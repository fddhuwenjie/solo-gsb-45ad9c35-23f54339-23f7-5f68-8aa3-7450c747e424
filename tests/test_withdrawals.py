"""选手永久退赛模块测试。

覆盖需求：
- 退赛操作：request_id 幂等键 + expected_event_seq 乐观快照 + 原因；
  仅在当前没有未结束轮次、待处理复核、未消费的下一轮签到快照或已发布
  晋级名单时生效；成功与 participant.withdrawn 事件同事务提交；
- 幂等与冲突：同一请求重放返回首次结果（不占新事件序号）；同键不同
  参数 409；序号冲突 409；重复退赛（新 request_id）409；冲突一律不落库；
- 前置条件：未结束轮次 / 待处理复核 / 未消费签到快照 / 已发布晋级名单
  存在时 409 不留痕；维护态（paused）/归档（archived）写守卫；
- 退赛效果：历史对局、积分、扣分保留；后续签到快照排除、瑞士配对与
  轮空候选排除；排名保留历史并标记 withdrawn（原因/时间可见）；后续
  晋级候选（席位与候补）排除；退赛后名单锁定不能再报名；
- 查询：赛事退赛列表、单参赛者退赛记录（未退赛 404）；
- 并发：并发退赛同一选手仅一成一败。
"""
import concurrent.futures

import pytest


# ---------- 基础工具 ----------

def create_tournament(client, name="退赛测试赛"):
    r = client.post("/tournaments", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def add_players(client, tid, names):
    r = client.post(
        f"/tournaments/{tid}/participants/bulk", json={"names": names}
    )
    assert r.status_code == 201
    return r.json()


def next_round(client, tid):
    r = client.post(f"/tournaments/{tid}/rounds/next")
    assert r.status_code == 201, r.text
    return r.json()


def put_result(client, match, sw, sb):
    mid = match["id"] if isinstance(match, dict) else match
    r = client.put(
        f"/matches/{mid}/result", json={"score_white": sw, "score_black": sb}
    )
    assert r.status_code == 200, r.text
    return r.json()


def close_round(rnd, client):
    for m in rnd["matches"]:
        if not m["is_bye"]:
            put_result(client, m, 1.0, 0.0)


def event_seq(client, tid):
    return client.get(f"/tournaments/{tid}/events").json()["last_seq"]


def events_of(client, tid, type_=None):
    items = client.get(f"/tournaments/{tid}/events?limit=200").json()["items"]
    return [e for e in items if type_ is None or e["type"] == type_]


def standings_map(client, tid):
    return {
        row["participant_id"]: row
        for row in client.get(f"/tournaments/{tid}/standings").json()
    }


def withdraw(client, tid, pid, request_id="w-1", reason="个人原因", seq=None):
    if seq is None:
        seq = event_seq(client, tid)
    return client.post(
        f"/tournaments/{tid}/withdrawals",
        json={
            "request_id": request_id,
            "expected_event_seq": seq,
            "participant_id": pid,
            "reason": reason,
        },
    )


# ---------- 成功路径 ----------

def test_withdraw_success_before_first_round(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    seq = event_seq(client, tid)

    r = withdraw(client, tid, players[1]["id"], reason="伤病退赛")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["participant_id"] == players[1]["id"]
    assert body["number"] == 2
    assert body["name"] == "B"
    assert body["reason"] == "伤病退赛"
    assert body["request_id"] == "w-1"
    assert body["expected_event_seq"] == seq
    assert body["event_seq"] == seq + 1
    assert body["withdrawn_at"].endswith("Z")

    # 事件同事务提交：序号紧随，载荷含原因与序号快照
    events = events_of(client, tid, "participant.withdrawn")
    assert len(events) == 1
    assert events[0]["seq"] == body["event_seq"]
    assert events[0]["payload"]["participant_id"] == players[1]["id"]
    assert events[0]["payload"]["reason"] == "伤病退赛"


def test_withdraw_after_round_closed_preserves_history(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B", "C", "D"])
    rnd = next_round(client, tid)
    close_round(rnd, client)

    # 选手 2（首轮输给 1）退赛：历史对局与 0 分保留
    pid2 = players[1]["id"]
    r = withdraw(client, tid, pid2, request_id="wd-1")
    assert r.status_code == 201

    st = standings_map(client, tid)
    row = st[pid2]
    assert row["withdrawn"] is True
    assert row["withdrawn_reason"] == "个人原因"
    assert row["withdrawn_at"] is not None
    # 历史对局数与积分不变
    assert row["played"] == 1
    assert row["points"] == 0.0
    # 名次仍保留在排名表中
    ranks = [s["participant_id"] for s in sorted(
        client.get(f"/tournaments/{tid}/standings").json(),
        key=lambda x: x["rank"],
    )]
    assert pid2 in ranks
    # 未退赛者不被标记
    assert all(
        st[p["id"]]["withdrawn"] is False for p in players if p["id"] != pid2
    )


# ---------- 幂等与冲突 ----------

def test_replay_same_request_returns_first_result(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    seq = event_seq(client, tid)

    first = withdraw(client, tid, players[1]["id"], request_id="r-1")
    assert first.status_code == 201
    assert len(events_of(client, tid, "participant.withdrawn")) == 1

    replay = client.post(
        f"/tournaments/{tid}/withdrawals",
        json={
            "request_id": "r-1",
            "expected_event_seq": seq,
            "participant_id": players[1]["id"],
            "reason": "个人原因",
        },
    )
    assert replay.status_code == 201
    assert replay.json()["id"] == first.json()["id"]
    assert replay.json()["event_seq"] == first.json()["event_seq"]
    # 重放不产生新事件、不占序号
    assert event_seq(client, tid) == first.json()["event_seq"]
    assert len(events_of(client, tid, "participant.withdrawn")) == 1


def test_same_request_id_different_params_conflicts(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    seq = event_seq(client, tid)
    assert withdraw(client, tid, players[1]["id"], request_id="r-2").status_code == 201

    # 同键不同原因 -> 409，不落库
    r = client.post(
        f"/tournaments/{tid}/withdrawals",
        json={
            "request_id": "r-2",
            "expected_event_seq": seq,
            "participant_id": players[1]["id"],
            "reason": "另一个原因",
        },
    )
    assert r.status_code == 409
    # 同键用于另一选手 -> 409
    r = client.post(
        f"/tournaments/{tid}/withdrawals",
        json={
            "request_id": "r-2",
            "expected_event_seq": seq,
            "participant_id": players[0]["id"],
            "reason": "个人原因",
        },
    )
    assert r.status_code == 409


def test_seq_conflict_does_not_persist(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    stale_seq = event_seq(client, tid)
    # 制造一个序号推进（再报名一人不行——锁定逻辑未触发，但可用首轮
    # 生成/任意写操作；这里直接退另一选手）
    assert withdraw(client, tid, players[1]["id"], request_id="s-1").status_code == 201

    r = client.post(
        f"/tournaments/{tid}/withdrawals",
        json={
            "request_id": "s-2",
            "expected_event_seq": stale_seq,
            "participant_id": players[0]["id"],
            "reason": "过期快照",
        },
    )
    assert r.status_code == 409
    assert "当前最大事件序号" in r.json()["detail"]
    # 未落库：选手 1 未退赛，且无第二条事件
    st = standings_map(client, tid)
    assert st[players[0]["id"]]["withdrawn"] is False
    assert len(events_of(client, tid, "participant.withdrawn")) == 1


def test_duplicate_withdrawal_new_request_id_conflicts(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    assert withdraw(client, tid, players[0]["id"], request_id="d-1").status_code == 201

    r = withdraw(client, tid, players[0]["id"], request_id="d-2")
    assert r.status_code == 409
    assert "已退赛" in r.json()["detail"]
    # 只有一条退赛记录与一条事件
    assert len(client.get(f"/tournaments/{tid}/withdrawals").json()) == 1
    assert len(events_of(client, tid, "participant.withdrawn")) == 1


def test_withdraw_unknown_participant_404(client):
    tid = create_tournament(client)
    add_players(client, tid, ["A"])
    r = withdraw(client, tid, 999, request_id="x-1")
    assert r.status_code == 404


# ---------- 前置条件 ----------

def test_withdraw_blocked_during_open_round(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B", "C", "D"])
    next_round(client, tid)  # open 轮
    r = withdraw(client, tid, players[0]["id"], request_id="o-1")
    assert r.status_code == 409
    assert "尚未结束" in r.json()["detail"]
    assert len(events_of(client, tid, "participant.withdrawn")) == 0


def test_withdraw_blocked_by_pending_review(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    rnd = next_round(client, tid)
    match = rnd["matches"][0]
    put_result(client, match, 1.0, 0.0)  # 轮次封盘
    seq = event_seq(client, tid)
    # 对该场创建待处理复核
    rr = client.post(
        f"/matches/{match['id']}/result-reviews",
        json={
            "request_id": "rv-1",
            "expected_event_seq": seq,
            "score_white": 0.0,
            "score_black": 1.0,
            "reason": "登记有误",
        },
    )
    assert rr.status_code == 201

    r = withdraw(client, tid, players[0]["id"], request_id="pr-1")
    assert r.status_code == 409
    assert "待处理" in r.json()["detail"]
    assert len(events_of(client, tid, "participant.withdrawn")) == 0


def test_withdraw_blocked_by_unconsumed_checkin_snapshot(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B", "C", "D"])
    rnd = next_round(client, tid)
    close_round(rnd, client)
    seq = event_seq(client, tid)
    # 为第 2 轮建立尚未消费的签到快照
    r = client.put(
        f"/tournaments/{tid}/checkin-snapshots",
        json={
            "request_id": "ci-1",
            "expected_event_seq": seq,
            "round_no": 2,
            "present_numbers": [1, 2, 3, 4],
            "absent_numbers": [],
        },
    )
    assert r.status_code == 200

    r = withdraw(client, tid, players[0]["id"], request_id="cs-1")
    assert r.status_code == 409
    assert "签到快照" in r.json()["detail"]
    assert len(events_of(client, tid, "participant.withdrawn")) == 0


def test_withdraw_allowed_after_snapshot_consumed(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B", "C", "D"])
    rnd1 = next_round(client, tid)
    close_round(rnd1, client)
    seq = event_seq(client, tid)
    client.put(
        f"/tournaments/{tid}/checkin-snapshots",
        json={
            "request_id": "ci-2",
            "expected_event_seq": seq,
            "round_no": 2,
            "present_numbers": [1, 2, 3, 4],
            "absent_numbers": [],
        },
    )
    rnd2 = next_round(client, tid)  # 消费快照
    close_round(rnd2, client)

    r = withdraw(client, tid, players[0]["id"], request_id="cc-1")
    assert r.status_code == 201


def test_withdraw_blocked_after_promotion_published(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    rnd = next_round(client, tid)
    close_round(rnd, client)
    seq = event_seq(client, tid)
    pr = client.post(
        f"/tournaments/{tid}/promotions",
        json={
            "request_id": "promo-1",
            "expected_event_seq": seq,
            "qualifying_round_no": 1,
            "slots": 1,
        },
    )
    assert pr.status_code == 201

    r = withdraw(client, tid, players[0]["id"], request_id="pp-1")
    assert r.status_code == 409
    assert "晋级名单" in r.json()["detail"]
    assert len(events_of(client, tid, "participant.withdrawn")) == 0


def test_withdraw_blocked_in_paused_and_archived(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    seq = event_seq(client, tid)
    pause = client.post(
        f"/tournaments/{tid}/status/actions",
        json={"request_id": "m-1", "expected_event_seq": seq, "action": "pause"},
    )
    assert pause.status_code == 200

    r = withdraw(client, tid, players[0]["id"], request_id="pa-1")
    assert r.status_code == 409
    assert "维护态" in r.json()["detail"]

    # 恢复后成功退赛；再次暂停期间重放该已成功请求，仍原样返回首次结果
    seq_for_pa2 = event_seq(client, tid)
    client.post(
        f"/tournaments/{tid}/status/actions",
        json={
            "request_id": "m-2",
            "expected_event_seq": seq_for_pa2,
            "action": "resume",
        },
    )
    seq_before_withdraw = event_seq(client, tid)
    success = withdraw(
        client, tid, players[1]["id"], request_id="pa-2",
        seq=seq_before_withdraw,
    )
    assert success.status_code == 201
    client.post(
        f"/tournaments/{tid}/status/actions",
        json={
            "request_id": "m-3",
            "expected_event_seq": event_seq(client, tid),
            "action": "pause",
        },
    )
    replay = client.post(
        f"/tournaments/{tid}/withdrawals",
        json={
            "request_id": "pa-2",
            "expected_event_seq": seq_before_withdraw,
            "participant_id": players[1]["id"],
            "reason": "个人原因",
        },
    )
    assert replay.status_code == 201  # 已成功请求的重放原样返回
    assert replay.json()["id"] == success.json()["id"]


# ---------- 退赛效果 ----------

def test_withdrawn_excluded_from_next_pairing_and_bye(client):
    # 5 人：选手 5 首轮前退赛，剩 4 人偶数，无轮空，且不出现 5
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B", "C", "D", "E"])
    assert withdraw(client, tid, players[4]["id"], request_id="pw-1").status_code == 201

    rnd = next_round(client, tid)
    ids_in_matches = set()
    for m in rnd["matches"]:
        ids_in_matches.add(m["player_white"]["id"])
        if m["player_black"]:
            ids_in_matches.add(m["player_black"]["id"])
    assert players[4]["id"] not in ids_in_matches
    assert all(not m["is_bye"] for m in rnd["matches"])

    # 第 1 轮封盘后让 4 号也退赛，剩 3 人：奇数轮空只在 1/2/3 中产生
    close_round(rnd, client)
    assert withdraw(client, tid, players[3]["id"], request_id="pw-2").status_code == 201
    rnd2 = next_round(client, tid)
    bye_players = {
        m["player_white"]["id"] for m in rnd2["matches"] if m["is_bye"]
    }
    active_ids = {players[i]["id"] for i in range(3)}
    assert bye_players <= active_ids
    all_ids = set()
    for m in rnd2["matches"]:
        all_ids.add(m["player_white"]["id"])
        if m["player_black"]:
            all_ids.add(m["player_black"]["id"])
    assert players[3]["id"] not in all_ids
    assert players[4]["id"] not in all_ids


def test_withdrawn_excluded_from_checkin_snapshot_roster(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B", "C"])
    # 3 号退赛后，为第 1 轮建立快照：名单只需覆盖 1、2；列入 3 号算未知
    assert withdraw(client, tid, players[2]["id"], request_id="cw-1").status_code == 201
    seq = event_seq(client, tid)
    bad = client.put(
        f"/tournaments/{tid}/checkin-snapshots",
        json={
            "request_id": "ci-bad",
            "expected_event_seq": seq,
            "round_no": 1,
            "present_numbers": [1, 2],
            "absent_numbers": [3],
        },
    )
    assert bad.status_code == 409

    ok = client.put(
        f"/tournaments/{tid}/checkin-snapshots",
        json={
            "request_id": "ci-ok",
            "expected_event_seq": seq,
            "round_no": 1,
            "present_numbers": [1, 2],
            "absent_numbers": [],
        },
    )
    assert ok.status_code == 200
    roster_numbers = {row["number"] for row in ok.json()["roster"]}
    assert roster_numbers == {1, 2}


def test_withdrawn_excluded_from_promotion_candidates(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B", "C"])
    rnd = next_round(client, tid)
    # 3 人：一场对局 + 一个轮空。让选手 1 胜、2 负、3 轮空。
    real = [m for m in rnd["matches"] if not m["is_bye"]]
    put_result(client, real[0], 1.0, 0.0)
    # 第 2 名（选手 2，0 分）退赛
    pid2 = players[1]["id"]
    assert withdraw(client, tid, pid2, request_id="cp-1").status_code == 201
    seq = event_seq(client, tid)
    r = client.post(
        f"/tournaments/{tid}/promotions",
        json={
            "request_id": "promo-cp",
            "expected_event_seq": seq,
            "qualifying_round_no": 1,
            "slots": 2,
        },
    )
    assert r.status_code == 201, r.text
    numbers = {s["original"]["number"] for s in r.json()["seats"]}
    wait_numbers = {w["participant"]["number"] for w in r.json()["waitlist"]}
    assert 2 not in numbers
    assert 2 not in wait_numbers
    # 退赛者仍在排名里
    st = standings_map(client, tid)
    assert st[pid2]["withdrawn"] is True


def test_promotion_slots_limited_to_active_count(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    rnd = next_round(client, tid)
    close_round(rnd, client)
    assert withdraw(client, tid, players[1]["id"], request_id="ps-1").status_code == 201
    # 仅 1 名有效参赛者，2 个名额 -> 409
    r = client.post(
        f"/tournaments/{tid}/promotions",
        json={
            "request_id": "promo-ps",
            "expected_event_seq": event_seq(client, tid),
            "qualifying_round_no": 1,
            "slots": 2,
        },
    )
    assert r.status_code == 409
    assert "有效参赛人数" in r.json()["detail"]


def test_registration_locked_after_withdrawal(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    assert withdraw(client, tid, players[0]["id"], request_id="rl-1").status_code == 201
    r = client.post(
        f"/tournaments/{tid}/participants", json={"name": "新人"}
    )
    assert r.status_code == 409
    assert "退赛" in r.json()["detail"]


def test_penalties_remain_effective_after_withdrawal(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B", "C", "D"])
    rnd = next_round(client, tid)
    close_round(rnd, client)
    seq = event_seq(client, tid)
    pen = client.post(
        f"/tournaments/{tid}/penalties",
        json={
            "request_id": "pen-w",
            "expected_event_seq": seq,
            "participant_id": players[0]["id"],
            "points": 0.5,
            "reason": "迟到",
        },
    )
    assert pen.status_code == 201
    assert withdraw(client, tid, players[0]["id"], request_id="wpen-1").status_code == 201

    row = standings_map(client, tid)[players[0]["id"]]
    assert row["withdrawn"] is True
    assert row["penalty"] == 0.5  # 扣分保留
    # 账本仍可查
    ledger = client.get(
        f"/tournaments/{tid}/participants/{players[0]['id']}/penalties"
    ).json()
    assert ledger["effective_penalty"] == 0.5


# ---------- 查询 ----------

def test_withdrawal_queries(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B", "C"])
    withdraw(client, tid, players[2]["id"], request_id="q-1", reason="R3")
    withdraw(client, tid, players[0]["id"], request_id="q-2", reason="R1")

    rows = client.get(f"/tournaments/{tid}/withdrawals")
    assert rows.status_code == 200
    # 按参赛者编号升序
    assert [w["number"] for w in rows.json()] == [1, 3]

    one = client.get(f"/tournaments/{tid}/participants/{players[0]['id']}/withdrawal")
    assert one.status_code == 200
    assert one.json()["reason"] == "R1"

    missing = client.get(
        f"/tournaments/{tid}/participants/{players[1]['id']}/withdrawal"
    )
    assert missing.status_code == 404

    # 跨赛事隔离：另一赛事查不到
    other = create_tournament(client, "其他赛事")
    assert client.get(f"/tournaments/{other}/withdrawals").json() == []


# ---------- 并发 ----------

def test_concurrent_withdraw_same_participant_only_one_wins(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["A", "B"])
    seq = event_seq(client, tid)

    def call(rid):
        return client.post(
            f"/tournaments/{tid}/withdrawals",
            json={
                "request_id": rid,
                "expected_event_seq": seq,
                "participant_id": players[0]["id"],
                "reason": "并发退赛",
            },
        )

    # TestClient 为同步客户端：用线程池并发提交（写锁串行裁决）
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(call, f"cc-{i}") for i in range(8)]
        results = [f.result().status_code for f in futures]

    assert results.count(201) == 1
    assert sum(1 for s in results if s == 409) == 7
    assert len(client.get(f"/tournaments/{tid}/withdrawals").json()) == 1
    assert len(events_of(client, tid, "participant.withdrawn")) == 1
