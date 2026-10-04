"""选手永久退赛模块测试。

覆盖需求：
- 退赛：request_id 幂等键 + expected_event_seq 乐观快照 + 理由；
  仅在没有未结束轮次、待处理赛果复核、未消费的下一轮签到快照或已发布
  晋级名单时生效，否则 409 且不落库；
- 幂等：重复请求（参数一致）原样返回首次结果、不产生新事件；同键不同
  参数 409；重复退赛（不同 request_id）409 不落库不占号；
- 数据保留与排除：历史对局、积分、扣分与对手 Buchholz 贡献保留；
  退赛者从后续签到快照、瑞士配对、轮空候选与晋级候选中排除；
- 查询：排名保留历史数据并标记 withdrawn（含理由/时间），参赛者名单
  带退赛字段，退赛列表可查；
- 事件：participant.withdrawn 与退赛状态同事务提交，被拒绝请求不产生
  事件、不占序号；
- 维护态：paused/archived 下退赛请求统一 409 不留痕；
- 并发：并发退赛同一选手仅一成一败。
"""
import concurrent.futures
import threading

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
        f"/matches/{mid}/result",
        json={"score_white": sw, "score_black": sb},
    )
    assert r.status_code == 200, r.text
    return r.json()


def event_seq(client, tid):
    return client.get(f"/tournaments/{tid}/events").json()["last_seq"]


def events_of(client, tid, type_=None):
    items = client.get(f"/tournaments/{tid}/events?limit=200").json()["items"]
    return [e for e in items if type_ is None or e["type"] == type_]


def standings_map(client, tid):
    rows = client.get(f"/tournaments/{tid}/standings").json()
    return {row["participant_id"]: row for row in rows}


def participants_map(client, tid):
    rows = client.get(f"/tournaments/{tid}/participants").json()
    return {row["id"]: row for row in rows}


def withdraw(client, tid, **body):
    return client.post(f"/tournaments/{tid}/withdrawals", json=body)


def withdraw_body(client, tid, participant_id, reason="伤病退赛",
                  request_id="wd-1", expected=None):
    return {
        "request_id": request_id,
        "expected_event_seq": (
            event_seq(client, tid) if expected is None else expected
        ),
        "participant_id": participant_id,
        "reason": reason,
    }


def close_round(client, rnd):
    real = [m for m in rnd["matches"] if not m["is_bye"]]
    for m in real:
        put_result(client, m, 1.0, 0.0)
    return real


def setup_closed_round(client, n=5):
    """n 人赛事第 1 轮全部白胜并封盘。返回 (tid, players, real)。"""
    tid = create_tournament(client)
    players = add_players(client, tid, [f"选手{i}" for i in range(1, n + 1)])
    rnd = next_round(client, tid)
    real = close_round(client, rnd)
    return tid, players, real


# ---------- 一、成功退赛 ----------

def test_withdraw_success_sets_status_event_and_response(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["甲", "乙"])
    seq = event_seq(client, tid)

    r = withdraw(client, tid, **withdraw_body(client, tid, players[1]["id"]))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["tournament_id"] == tid
    assert body["participant_id"] == players[1]["id"]
    assert body["number"] == 2
    assert body["name"] == "乙"
    assert body["reason"] == "伤病退赛"
    assert body["request_id"] == "wd-1"
    assert body["expected_event_seq"] == seq
    # 事件与状态同事务：序号恰好 +1
    assert body["event_seq"] == seq + 1
    assert body["withdrawn_at"].endswith("Z")

    assert event_seq(client, tid) == seq + 1
    evs = events_of(client, tid, "participant.withdrawn")
    assert len(evs) == 1
    assert evs[0]["seq"] == seq + 1
    assert evs[0]["payload"]["participant_id"] == players[1]["id"]
    assert evs[0]["payload"]["number"] == 2
    assert evs[0]["payload"]["reason"] == "伤病退赛"

    p = participants_map(client, tid)[players[1]["id"]]
    assert p["withdrawn"] is True
    assert p["withdrawn_reason"] == "伤病退赛"
    assert p["withdrawn_at"] == body["withdrawn_at"]
    # 未退赛者字段为空
    assert participants_map(client, tid)[players[0]["id"]]["withdrawn"] is False


def test_withdraw_allowed_between_rounds(client):
    tid, players, real = setup_closed_round(client, n=4)
    seq = event_seq(client, tid)
    r = withdraw(client, tid, **withdraw_body(client, tid, players[2]["id"]))
    assert r.status_code == 201, r.text
    assert event_seq(client, tid) == seq + 1


def test_withdraw_allowed_before_any_round(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["甲", "乙"])
    r = withdraw(client, tid, **withdraw_body(client, tid, players[0]["id"]))
    assert r.status_code == 201, r.text


# ---------- 二、前置条件冲突 ----------

def test_withdraw_blocked_while_round_open(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["甲", "乙", "丙", "丁"])
    next_round(client, tid)  # open
    seq = event_seq(client, tid)

    r = withdraw(client, tid, **withdraw_body(client, tid, players[0]["id"]))
    assert r.status_code == 409
    assert "尚未结束" in r.json()["detail"]
    assert event_seq(client, tid) == seq  # 不落库、不占号


def test_withdraw_blocked_by_pending_review(client):
    tid, players, real = setup_closed_round(client)
    client.post(
        f"/matches/{real[0]['id']}/result-reviews",
        json={
            "request_id": "rev-1",
            "expected_event_seq": event_seq(client, tid),
            "score_white": 0.0,
            "score_black": 1.0,
            "reason": "记分表登记有误",
        },
    )
    seq = event_seq(client, tid)

    r = withdraw(client, tid, **withdraw_body(client, tid, players[3]["id"]))
    assert r.status_code == 409
    assert "复核" in r.json()["detail"]
    assert event_seq(client, tid) == seq


def test_withdraw_blocked_by_unconsumed_checkin_snapshot(client):
    tid, players, real = setup_closed_round(client)
    # 为第 2 轮建立未消费签到快照
    sr = client.put(
        f"/tournaments/{tid}/checkin-snapshots",
        json={
            "request_id": "ci-1",
            "expected_event_seq": event_seq(client, tid),
            "round_no": 2,
            "present_numbers": [p["number"] for p in players],
            "absent_numbers": [],
        },
    )
    assert sr.status_code == 200
    seq = event_seq(client, tid)

    r = withdraw(client, tid, **withdraw_body(client, tid, players[3]["id"]))
    assert r.status_code == 409
    assert "签到快照" in r.json()["detail"]
    assert event_seq(client, tid) == seq
    # 快照仍可用于后续生成（未被退赛请求改动）
    rnd = next_round(client, tid)
    assert rnd["round_no"] == 2


def test_withdraw_allowed_after_snapshot_consumed(client):
    tid, players, real = setup_closed_round(client)
    client.put(
        f"/tournaments/{tid}/checkin-snapshots",
        json={
            "request_id": "ci-1",
            "expected_event_seq": event_seq(client, tid),
            "round_no": 2,
            "present_numbers": [p["number"] for p in players],
            "absent_numbers": [],
        },
    )
    close_round(client, next_round(client, tid))

    r = withdraw(client, tid, **withdraw_body(client, tid, players[3]["id"]))
    assert r.status_code == 201, r.text


def test_withdraw_blocked_after_promotion_published(client):
    tid, players, real = setup_closed_round(client, n=4)
    pr = client.post(
        f"/tournaments/{tid}/promotions",
        json={
            "request_id": "promo-1",
            "expected_event_seq": event_seq(client, tid),
            "qualifying_round_no": 1,
            "slots": 2,
        },
    )
    assert pr.status_code == 201
    seq = event_seq(client, tid)

    r = withdraw(client, tid, **withdraw_body(client, tid, players[0]["id"]))
    assert r.status_code == 409
    assert "晋级名单" in r.json()["detail"]
    assert event_seq(client, tid) == seq


def test_withdraw_snapshot_conflict_rejects_without_write(client):
    tid, players, real = setup_closed_round(client)
    seq = event_seq(client, tid)
    r = withdraw(
        client,
        tid,
        **withdraw_body(client, tid, players[0]["id"], expected=seq - 1),
    )
    assert r.status_code == 409
    assert "快照" in r.json()["detail"]
    assert event_seq(client, tid) == seq
    assert participants_map(client, tid)[players[0]["id"]]["withdrawn"] is False
    assert client.get(f"/tournaments/{tid}/withdrawals").json() == []


def test_withdraw_participant_not_found(client):
    tid, players, real = setup_closed_round(client)
    r = withdraw(client, tid, **withdraw_body(client, tid, 9999))
    assert r.status_code == 404
    # 赛事不存在
    r = withdraw(client, 9999, **withdraw_body(client, tid, 1))
    assert r.status_code == 404


def test_withdraw_validation_errors(client):
    tid, players, real = setup_closed_round(client)
    seq = event_seq(client, tid)
    # 空理由 -> 422
    r = withdraw(
        client, tid, **withdraw_body(client, tid, players[0]["id"], reason="")
    )
    assert r.status_code == 422
    assert event_seq(client, tid) == seq


# ---------- 三、幂等与重复退赛 ----------

def test_withdraw_idempotent_replay(client):
    tid, players, real = setup_closed_round(client)
    body = withdraw_body(client, tid, players[0]["id"])
    r1 = withdraw(client, tid, **body)
    assert r1.status_code == 201
    seq = event_seq(client, tid)

    r2 = withdraw(client, tid, **body)
    assert r2.status_code == 201
    assert r2.json() == r1.json()
    assert event_seq(client, tid) == seq  # 不产生新事件
    assert len(events_of(client, tid, "participant.withdrawn")) == 1
    assert len(client.get(f"/tournaments/{tid}/withdrawals").json()) == 1


def test_withdraw_same_request_id_different_params_conflict(client):
    tid, players, real = setup_closed_round(client)
    body = withdraw_body(client, tid, players[0]["id"])
    assert withdraw(client, tid, **body).status_code == 201
    seq = event_seq(client, tid)

    # 换参赛者
    r = withdraw(client, tid, **{**body, "participant_id": players[1]["id"]})
    assert r.status_code == 409
    # 换理由
    r = withdraw(client, tid, **{**body, "reason": "另一个理由"})
    assert r.status_code == 409
    assert event_seq(client, tid) == seq
    # 只有最初一位退赛
    assert participants_map(client, tid)[players[0]["id"]]["withdrawn"] is True
    assert participants_map(client, tid)[players[1]["id"]]["withdrawn"] is False
    assert len(client.get(f"/tournaments/{tid}/withdrawals").json()) == 1


def test_duplicate_withdraw_different_request_id_conflict(client):
    tid, players, real = setup_closed_round(client)
    assert withdraw(
        client, tid, **withdraw_body(client, tid, players[0]["id"])
    ).status_code == 201
    seq = event_seq(client, tid)

    r = withdraw(
        client,
        tid,
        **withdraw_body(
            client, tid, players[0]["id"], request_id="wd-2", reason="再退一次"
        ),
    )
    assert r.status_code == 409
    assert "已退赛" in r.json()["detail"]
    assert event_seq(client, tid) == seq
    # 理由保持首次值
    p = participants_map(client, tid)[players[0]["id"]]
    assert p["withdrawn_reason"] == "伤病退赛"


# ---------- 四、历史数据保留，后续编排排除 ----------

def test_withdrawn_keeps_history_marks_standings(client):
    # 第 1 轮：1>2、3>4、5 轮空
    tid, players, real = setup_closed_round(client, n=5)
    withdrawn = players[4]  # 5 号轮空得 1 分

    assert withdraw(
        client, tid, **withdraw_body(client, tid, withdrawn["id"])
    ).status_code == 201

    rows = standings_map(client, tid)
    row = rows[withdrawn["id"]]
    assert row["withdrawn"] is True
    assert row["withdrawn_reason"] == "伤病退赛"
    assert row["withdrawn_at"] is not None
    # 轮空历史积分保留
    assert row["points"] == 1.0
    assert row["byes"] == 1
    # 排名仍在列表中（历史数据保留）
    numbers = [r["number"] for r in
               client.get(f"/tournaments/{tid}/standings").json()]
    assert 5 in numbers


def test_withdrawn_excluded_from_next_round_pairing_and_bye(client):
    # 6 人第 1 轮 1>2、3>4、5>6；退赛 6 号后第 2 轮剩 5 人：
    # 6 号不得出现在任何场次，奇数人轮空从其余 5 人中选
    tid, players, real = setup_closed_round(client, n=6)
    pid6 = players[5]["id"]
    assert withdraw(client, tid, **withdraw_body(client, tid, pid6)).status_code == 201

    rnd = next_round(client, tid)
    involved = {
        pp["id"]
        for m in rnd["matches"]
        for pp in (m["player_white"], m["player_black"])
        if pp is not None
    }
    assert pid6 not in involved
    byes = [m for m in rnd["matches"] if m["is_bye"]]
    assert len(byes) == 1  # 5 人奇数
    assert byes[0]["player_white"]["id"] != pid6


def test_withdrawn_excluded_without_checkin_snapshot(client):
    # 无签到快照时退赛者同样被排除（全员 5 人，1 人退赛 -> 4 人无轮空）
    tid = create_tournament(client)
    players = add_players(client, tid, ["甲", "乙", "丙", "丁", "戊"])
    assert withdraw(
        client, tid, **withdraw_body(client, tid, players[4]["id"])
    ).status_code == 201

    rnd = next_round(client, tid)
    assert all(not m["is_bye"] for m in rnd["matches"])
    assert len(rnd["matches"]) == 2
    involved = {
        pp["number"]
        for m in rnd["matches"]
        for pp in (m["player_white"], m["player_black"])
        if pp is not None
    }
    assert involved == {1, 2, 3, 4}


def test_withdrawn_excluded_from_checkin_snapshot_coverage(client):
    tid = create_tournament(client)
    players = add_players(client, tid, ["甲", "乙", "丙", "丁", "戊"])
    pid5 = players[4]["id"]
    assert withdraw(client, tid, **withdraw_body(client, tid, pid5)).status_code == 201

    # 快照仍包含退赛者 -> 409（视为未知编号）
    r = client.put(
        f"/tournaments/{tid}/checkin-snapshots",
        json={
            "request_id": "ci-1",
            "expected_event_seq": event_seq(client, tid),
            "round_no": 1,
            "present_numbers": [1, 2, 3, 4, 5],
            "absent_numbers": [],
        },
    )
    assert r.status_code == 409

    # 只覆盖未退赛 4 人 -> 成功，roster 不含退赛者
    r = client.put(
        f"/tournaments/{tid}/checkin-snapshots",
        json={
            "request_id": "ci-2",
            "expected_event_seq": event_seq(client, tid),
            "round_no": 1,
            "present_numbers": [1, 2, 3, 4],
            "absent_numbers": [],
        },
    )
    assert r.status_code == 200
    assert {x["number"] for x in r.json()["roster"]} == {1, 2, 3, 4}


def test_withdrawn_buchholz_still_contributes_to_opponents(client):
    # 5 号第 1 轮告负后退赛：其对手（胜者）历史 Buchholz 不受退赛影响；
    # 5 号自身的历史数据也保留
    tid, players, real = setup_closed_round(client, n=6)
    pid6 = players[5]["id"]
    winner = real[2]["player_white"]["id"]  # 5 号

    assert withdraw(client, tid, **withdraw_body(client, tid, pid6)).status_code == 201
    rows = standings_map(client, tid)
    # 5 号对手是 6 号（0 分，退赛）：Buchholz 仍按 6 号当前排名积分计
    assert rows[winner]["buchholz"] == rows[pid6]["rank_points"] == 0.0
    # 6 号历史 0 分保留
    assert rows[pid6]["points"] == 0.0
    assert rows[pid6]["played"] == 1


def test_withdrawn_history_survives_later_rounds(client):
    tid, players, real = setup_closed_round(client, n=5)
    pid5 = players[4]["id"]
    assert withdraw(client, tid, **withdraw_body(client, tid, pid5)).status_code == 201

    # 第 2、3 轮正常进行（4 人偶数，无轮空）
    for _ in range(2):
        close_round(client, next_round(client, tid))

    row = standings_map(client, tid)[pid5]
    # 第 1 轮轮空 1 分始终保留，played/byes 不随后续轮次变化
    assert row["points"] == 1.0
    assert row["byes"] == 1
    assert row["played"] == 0
    assert row["withdrawn"] is True


# ---------- 五、晋级候选排除 ----------

def test_withdrawn_excluded_from_promotion_candidates(client):
    # 4 人第 1 轮封盘后退赛排名最高的 1 号；发布 2 个名额时冻结的
    # 席位/候补中都不应出现 1 号
    tid, players, real = setup_closed_round(client, n=4)
    pid1 = players[0]["id"]
    assert withdraw(client, tid, **withdraw_body(client, tid, pid1)).status_code == 201

    r = client.post(
        f"/tournaments/{tid}/promotions",
        json={
            "request_id": "promo-1",
            "expected_event_seq": event_seq(client, tid),
            "qualifying_round_no": 1,
            "slots": 3,
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    frozen = {s["original"]["participant_id"] for s in body["seats"]}
    frozen |= {w["participant"]["participant_id"] for w in body["waitlist"]}
    assert pid1 not in frozen
    assert len(frozen) == 3


def test_promotion_slots_capped_at_active_count(client):
    tid, players, real = setup_closed_round(client, n=4)
    assert withdraw(
        client, tid, **withdraw_body(client, tid, players[0]["id"])
    ).status_code == 201

    r = client.post(
        f"/tournaments/{tid}/promotions",
        json={
            "request_id": "promo-1",
            "expected_event_seq": event_seq(client, tid),
            "qualifying_round_no": 1,
            "slots": 4,
        },
    )
    assert r.status_code == 409
    assert "可晋级参赛人数" in r.json()["detail"]


# ---------- 六、扣分与退赛共存 ----------

def test_penalty_after_withdrawal_reflects_in_standings(client):
    tid, players, real = setup_closed_round(client, n=4)
    target = players[0]
    assert withdraw(
        client, tid, **withdraw_body(client, tid, target["id"])
    ).status_code == 201

    # 退赛后仍可登记扣分（轮次已结束、无待处理复核）
    r = client.post(
        f"/tournaments/{tid}/penalties",
        json={
            "request_id": "pen-1",
            "expected_event_seq": event_seq(client, tid),
            "participant_id": target["id"],
            "points": 0.5,
            "reason": "退赛前违规的追溯扣分",
        },
    )
    assert r.status_code == 201, r.text
    row = standings_map(client, tid)[target["id"]]
    assert row["withdrawn"] is True
    assert row["penalty"] == 0.5
    assert row["rank_points"] == 0.5  # 白胜 1 分 - 0.5


# ---------- 七、维护态与查询 ----------

def test_withdraw_blocked_in_paused_and_archived(client):
    tid, players, real = setup_closed_round(client)
    target = players[0]["id"]
    seq = event_seq(client, tid)
    client.post(
        f"/tournaments/{tid}/status/actions",
        json={"request_id": "pause-1",
              "expected_event_seq": seq, "action": "pause"},
    )
    r = withdraw(client, tid, **withdraw_body(client, tid, target))
    assert r.status_code == 409
    assert "维护态" in r.json()["detail"]

    # resume 后可退
    client.post(
        f"/tournaments/{tid}/status/actions",
        json={"request_id": "resume-1",
              "expected_event_seq": event_seq(client, tid), "action": "resume"},
    )
    assert withdraw(client, tid, **withdraw_body(client, tid, target)).status_code == 201

    # pause -> archive 后再退赛（另一选手）409
    cur = event_seq(client, tid)
    client.post(
        f"/tournaments/{tid}/status/actions",
        json={"request_id": "pause-2",
              "expected_event_seq": cur, "action": "pause"},
    )
    client.post(
        f"/tournaments/{tid}/status/actions",
        json={"request_id": "archive-1",
              "expected_event_seq": event_seq(client, tid),
              "action": "archive"},
    )
    r = withdraw(
        client, tid, **withdraw_body(client, tid, players[1]["id"],
                                     request_id="wd-arch")
    )
    assert r.status_code == 409
    assert "归档" in r.json()["detail"]


def test_withdrawals_list_query(client):
    tid, players, real = setup_closed_round(client, n=4)
    r1 = withdraw(client, tid, **withdraw_body(
        client, tid, players[0]["id"], reason="r1", request_id="wd-1"))
    assert r1.status_code == 201
    r2 = withdraw(client, tid, **withdraw_body(
        client, tid, players[1]["id"], reason="r2", request_id="wd-2"))
    assert r2.status_code == 201

    lst = client.get(f"/tournaments/{tid}/withdrawals")
    assert lst.status_code == 200
    bodies = lst.json()
    assert [b["participant_id"] for b in bodies] == [players[0]["id"],
                                                     players[1]["id"]]
    assert [b["reason"] for b in bodies] == ["r1", "r2"]
    assert bodies[0]["event_seq"] < bodies[1]["event_seq"]


# ---------- 八、并发 ----------

@pytest.fixture()
def live_server_factory():
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    from app.database import Base, engine
    from app.main import app

    def wait_until(cond, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return
            time.sleep(0.01)
        raise AssertionError("server not started")

    class Server:
        def __init__(self):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
            sock.close()
            self.server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=self.port,
                    log_level="error",
                    access_log=False,
                )
            )
            self.thread = threading.Thread(target=self.server.run, daemon=True)

        def client(self):
            return httpx.Client(
                base_url=f"http://127.0.0.1:{self.port}", timeout=10.0
            )

        def __enter__(self):
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            self.thread.start()
            wait_until(lambda: self.server.started)
            return self

        def __exit__(self, *exc):
            self.server.should_exit = True
            self.thread.join(timeout=10)

    return Server


def test_concurrent_withdraw_same_participant_only_one_succeeds(live_server_factory):
    """并发退赛同一位选手：恰好一条成功，其余 409。"""
    with live_server_factory() as srv:
        client = srv.client()
        tid, players, real = setup_closed_round(client)
        target = players[0]["id"]
        seq = event_seq(client, tid)
        n = 6
        barrier = threading.Barrier(n)

        def call(i):
            local = srv.client()
            try:
                barrier.wait(timeout=10)
                r = local.post(
                    f"/tournaments/{tid}/withdrawals",
                    json={
                        "request_id": f"wd-c-{i}",
                        "expected_event_seq": seq,
                        "participant_id": target,
                        "reason": f"理由{i}",
                    },
                )
                return r.status_code
            finally:
                local.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(call, i) for i in range(n)]
            results = [f.result(timeout=30) for f in futs]

        assert results.count(201) == 1
        assert results.count(409) == n - 1
        assert len(client.get(f"/tournaments/{tid}/withdrawals").json()) == 1
        assert event_seq(client, tid) == seq + 1  # 仅一次退赛占一个序号
        assert participants_map(client, tid)[target]["withdrawn"] is True


def test_concurrent_withdraw_same_key_different_participant(live_server_factory):
    """并发以同一 request_id 退赛不同选手：恰好一条落库，其余 409。"""
    with live_server_factory() as srv:
        client = srv.client()
        tid, players, real = setup_closed_round(client)
        seq = event_seq(client, tid)
        targets = [p["id"] for p in players]
        n = min(4, len(targets))
        barrier = threading.Barrier(n)

        def call(i):
            local = srv.client()
            try:
                barrier.wait(timeout=10)
                r = local.post(
                    f"/tournaments/{tid}/withdrawals",
                    json={
                        "request_id": "same-wd-key",
                        "expected_event_seq": seq,
                        "participant_id": targets[i],
                        "reason": "同键并发",
                    },
                )
                return r.status_code
            finally:
                local.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(call, i) for i in range(n)]
            results = [f.result(timeout=30) for f in futs]

        assert results.count(201) == 1
        assert results.count(409) == n - 1
        assert len(client.get(f"/tournaments/{tid}/withdrawals").json()) == 1
        assert event_seq(client, tid) == seq + 1
