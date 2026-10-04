"""选手永久退赛。

- withdraw_participant：为参赛者登记永久退赛。仅在以下条件全部满足时
  生效：没有未结束轮次、没有待处理赛果复核、没有未消费（尚未被轮次
  生成使用）的下一轮签到快照、没有已发布的晋级名单。成功时在同一写
  事务内把参赛者置为 withdrawn（记录理由与 UTC 时间）并写入
  participant.withdrawn 事件。
- 退赛保留全部历史对局、积分与扣分；compute_records 仍计算其历史
  成绩（排名查询中标记 withdrawn），而编排下一轮（services/rounds）、
  未消费签到快照名单（services/checkins）与晋级候选
  （services/promotions）统一排除退赛选手。
- 幂等：request_id 在赛事内唯一。重复请求参数一致时原样返回首次响应
  快照（不产生任何写入与事件）；同键不同参数返回 409。重复退赛（不同
  request_id）返回 409 且不落库。

注意：withdraw_participant 只在调用方事务内写入并 flush，不提交；
参赛者退赛状态、幂等记录与 participant.withdrawn 事件由路由层在同一
事务中统一提交。所有调用均处于进程级 writer_lock 内，并由
(tournament_id, request_id)、participant_id 唯一约束在数据库层兜底。
"""
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..events.service import PARTICIPANT_WITHDRAWN, append_event
from ..models import (
    CheckInSnapshot,
    Event,
    Participant,
    PromotionList,
    Round,
    WithdrawalAction,
)
from ..schemas import WithdrawalRequest, WithdrawalResponse
from .announcements import current_event_seq
from .lifecycle import ensure_tournament_writable
from .result_reviews import has_pending_review_for_tournament


def withdrawn_participant_ids(db: Session, tournament_id: int) -> set[int]:
    """赛事内已永久退赛的参赛者 id 集合。"""
    rows = (
        db.query(Participant.id)
        .filter(
            Participant.tournament_id == tournament_id,
            Participant.withdrawn.is_(True),
        )
        .all()
    )
    return {pid for (pid,) in rows}


def _find_replay(
    db: Session, tournament_id: int, req: WithdrawalRequest
) -> WithdrawalResponse | None:
    """同一 request_id 的已成功退赛：参数一致返回首次结果，否则 409。"""
    record = (
        db.query(WithdrawalAction)
        .filter(
            WithdrawalAction.tournament_id == tournament_id,
            WithdrawalAction.request_id == req.request_id,
        )
        .first()
    )
    if record is None:
        return None
    # 记录字段本身即参数指纹（participant/reason/expected_event_seq 不可变）
    same = (
        record.participant_id == req.participant_id
        and record.reason == req.reason
        and record.expected_event_seq == req.expected_event_seq
    )
    if same:
        return WithdrawalResponse(**record.response)
    raise HTTPException(
        status_code=409,
        detail="request_id 已被用于该赛事的其他退赛请求，请更换 request_id",
    )


def _ensure_withdrawable(db: Session, tournament_id: int) -> None:
    """退赛前置条件：无未结束轮次、待处理复核、未消费签到快照、已发布名单。"""
    open_round = (
        db.query(Round)
        .filter(
            Round.tournament_id == tournament_id,
            Round.status == "open",
        )
        .first()
    )
    if open_round is not None:
        raise HTTPException(
            status_code=409,
            detail=f"第 {open_round.round_no} 轮尚未结束，不能办理退赛",
        )
    if has_pending_review_for_tournament(db, tournament_id):
        raise HTTPException(
            status_code=409,
            detail="赛事存在待处理的赛果复核，需待复核关闭后才能办理退赛",
        )
    # 未消费的下一轮签到快照（目标轮次尚未生成）：快照名单仍含该选手
    unconsumed = (
        db.query(CheckInSnapshot)
        .filter(
            CheckInSnapshot.tournament_id == tournament_id,
            CheckInSnapshot.consumed.is_(False),
        )
        .first()
    )
    if unconsumed is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"存在第 {unconsumed.round_no} 轮未消费的签到快照，"
                "请先更新快照或生成该轮后再办理退赛"
            ),
        )
    published = (
        db.query(PromotionList.id)
        .filter(PromotionList.tournament_id == tournament_id)
        .first()
    )
    if published is not None:
        raise HTTPException(
            status_code=409,
            detail="赛事已发布晋级名单，不能再办理退赛",
        )


def withdraw_participant(
    db: Session, tournament_id: int, req: WithdrawalRequest
) -> tuple[WithdrawalResponse, Event | None]:
    """办理一次永久退赛，返回（响应, 事件）。

    幂等重放时返回（首次响应, None），不产生任何写入；资格不符或事件
    序号快照冲突抛 409，且在任何写入之前抛出（不落库）。调用方负责在
    同一事务中提交退赛状态、幂等记录与事件。
    """
    # 维护态/归档赛事拒绝一切写操作（在幂等判定与任何写入之前）
    ensure_tournament_writable(db, tournament_id)
    # 幂等判定先于业务校验：重试已成功请求必须返回首次结果
    replay = _find_replay(db, tournament_id, req)
    if replay is not None:
        return replay, None

    # 写锁可能经过等待才取得：以锁内已提交的最新状态重新校验
    participant = (
        db.query(Participant)
        .filter(
            Participant.id == req.participant_id,
            Participant.tournament_id == tournament_id,
        )
        .first()
    )
    if participant is None:
        raise HTTPException(status_code=404, detail="参赛者不存在")

    # 重复退赛（不同 request_id）拒绝且不落库
    if participant.withdrawn:
        raise HTTPException(
            status_code=409,
            detail=f"编号 {participant.number} 的参赛者已退赛，不能重复退赛",
        )

    _ensure_withdrawable(db, tournament_id)

    actual_seq = current_event_seq(db, tournament_id)
    if req.expected_event_seq != actual_seq:
        raise HTTPException(
            status_code=409,
            detail=(
                f"赛事快照已变化：当前最大事件序号为 {actual_seq}，"
                f"请求期望 {req.expected_event_seq}"
            ),
        )

    # 退赛状态与事件在同一事务写入：历史对局、积分与扣分原样保留
    now = datetime.utcnow()
    participant.withdrawn = True
    participant.withdrawn_reason = req.reason
    participant.withdrawn_at = now
    db.flush()

    event = append_event(
        db,
        tournament_id,
        PARTICIPANT_WITHDRAWN,
        {
            "participant_id": participant.id,
            "number": participant.number,
            "name": participant.name,
            "reason": req.reason,
            "expected_event_seq": req.expected_event_seq,
        },
    )

    response = WithdrawalResponse(
        tournament_id=tournament_id,
        participant_id=participant.id,
        number=participant.number,
        name=participant.name,
        reason=req.reason,
        request_id=req.request_id,
        expected_event_seq=req.expected_event_seq,
        event_seq=event.seq,
        withdrawn_at=now,
    )
    db.add(
        WithdrawalAction(
            tournament_id=tournament_id,
            participant_id=participant.id,
            request_id=req.request_id,
            expected_event_seq=req.expected_event_seq,
            reason=req.reason,
            response=response.model_dump(mode="json"),
        )
    )
    db.flush()
    return response, event
