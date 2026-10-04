"""选手永久退赛（withdrawal）。

- withdraw_participant：为参赛者登记永久退赛（终态，不可撤销）。请求
  携带 request_id 幂等键、expected_event_seq 乐观快照与退赛原因。
  仅在「当前没有未结束轮次、待处理赛果复核、未消费（尚未生成）的下一
  轮签到快照或已发布晋级名单」时生效。成功时在同一写事务内写入
  participant.withdrawn 事件。
- 退赛效果：历史对局、积分与扣分完整保留；选手从退赛之后的签到快照
  （名单恰好覆盖未退赛者）、瑞士配对与轮空候选中排除；排名查询保留其
  历史数据并标记 withdrawn（含原因与时间）；后续发布晋级名单时候选
  （席位与候补）排除该选手。
- 幂等：request_id 在赛事内唯一。重复请求参数一致时原样返回首次响应
  快照（不产生任何写入与事件）；同键不同参数返回 409。重复退赛（换用
  新 request_id）拒绝且不落库；每位选手至多一条退赛记录，由
  participant_id 唯一约束在数据库层兜底。

注意：本模块的写函数只在调用方事务内写入并 flush，不提交；退赛记录与
participant.withdrawn 事件由路由层在同一事务中统一提交。所有调用均处于
进程级 writer_lock 内。
"""
from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..events.service import PARTICIPANT_WITHDRAWN, append_event
from ..models import (
    CheckInSnapshot,
    Event,
    Participant,
    ParticipantWithdrawal,
    PromotionList,
    Round,
)
from ..schemas import WithdrawalRequest, WithdrawalResponse
from .announcements import current_event_seq
from .lifecycle import ensure_tournament_writable
from .result_reviews import has_pending_review_for_tournament


def withdrawals_map(db: Session, tournament_id: int) -> dict[int, ParticipantWithdrawal]:
    """赛事内各参赛者的退赛记录（未退赛者不在字典中）。"""
    rows = (
        db.query(ParticipantWithdrawal)
        .filter(ParticipantWithdrawal.tournament_id == tournament_id)
        .all()
    )
    return {w.participant_id: w for w in rows}


def get_withdrawal(
    db: Session, tournament_id: int, participant_id: int
) -> ParticipantWithdrawal | None:
    return (
        db.query(ParticipantWithdrawal)
        .filter(
            ParticipantWithdrawal.tournament_id == tournament_id,
            ParticipantWithdrawal.participant_id == participant_id,
        )
        .first()
    )


def list_withdrawals(
    db: Session, tournament_id: int
) -> list[ParticipantWithdrawal]:
    """赛事下的全部退赛记录（按参赛者编号升序）。"""
    return list(
        db.query(ParticipantWithdrawal)
        .join(Participant, Participant.id == ParticipantWithdrawal.participant_id)
        .filter(ParticipantWithdrawal.tournament_id == tournament_id)
        .order_by(Participant.number)
        .all()
    )


def serialize_withdrawal(
    withdrawal: ParticipantWithdrawal, participant: Participant
) -> WithdrawalResponse:
    return WithdrawalResponse(
        id=withdrawal.id,
        tournament_id=withdrawal.tournament_id,
        participant_id=participant.id,
        number=participant.number,
        name=participant.name,
        reason=withdrawal.reason,
        request_id=withdrawal.request_id,
        expected_event_seq=withdrawal.expected_event_seq,
        event_seq=withdrawal.response["event_seq"],
        withdrawn_at=withdrawal.created_at,
    )


def _fingerprint(req: WithdrawalRequest) -> dict:
    return {
        "participant_id": req.participant_id,
        "reason": req.reason,
        "expected_event_seq": req.expected_event_seq,
    }


def _find_replay(
    db: Session, tournament_id: int, req: WithdrawalRequest
) -> WithdrawalResponse | None:
    """同 request_id 的已成功请求：参数一致返回首次响应，否则 409。"""
    record = (
        db.query(ParticipantWithdrawal)
        .filter(
            ParticipantWithdrawal.tournament_id == tournament_id,
            ParticipantWithdrawal.request_id == req.request_id,
        )
        .first()
    )
    if record is None:
        return None
    if record.params == _fingerprint(req):
        return WithdrawalResponse(**record.response)
    raise HTTPException(
        status_code=409,
        detail="request_id 已被用于该赛事的其他退赛请求，请更换 request_id",
    )


def _ensure_withdrawable(db: Session, tournament_id: int) -> None:
    """退赛前置条件：无未结束轮次、无待处理复核、无未消费的下一轮
    签到快照、无已发布晋级名单。"""
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
    # 存在为「尚未生成的下一轮」准备的签到快照时拒绝：退赛会改变
    # 报名名单覆盖口径，运营方应先删除/重建快照（快照被轮次消费后
    # consumed=True，不再阻止退赛）
    pending_snapshot = (
        db.query(CheckInSnapshot)
        .filter(
            CheckInSnapshot.tournament_id == tournament_id,
            CheckInSnapshot.consumed.is_(False),
        )
        .first()
    )
    if pending_snapshot is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"第 {pending_snapshot.round_no} 轮存在尚未消费的签到快照，"
                "请先处理该快照后再办理退赛"
            ),
        )
    # 晋级名单一经发布即冻结候选口径，之后不再允许退赛
    published = (
        db.query(PromotionList.id)
        .filter(PromotionList.tournament_id == tournament_id)
        .first()
    )
    if published is not None:
        raise HTTPException(
            status_code=409,
            detail="赛事已发布晋级名单，名单冻结后不能再办理退赛",
        )


def withdraw_participant(
    db: Session, tournament_id: int, req: WithdrawalRequest
) -> tuple[WithdrawalResponse, Event | None]:
    """办理一次永久退赛（flush，不提交），返回（响应, 事件）。

    幂等重放时返回（首次响应, None），不产生任何写入；资格不符、重复
    退赛或事件序号快照冲突抛 409，且在任何写入之前抛出（不落库）。
    """
    # 幂等判定先于一切业务校验（含维护态守卫）：网络重试已成功请求
    # 必须在任何赛事状态下都返回首次结果
    replay = _find_replay(db, tournament_id, req)
    if replay is not None:
        return replay, None

    # 维护态/归档赛事拒绝新的写操作（在任何写入之前）
    ensure_tournament_writable(db, tournament_id)

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

    if get_withdrawal(db, tournament_id, participant.id) is not None:
        raise HTTPException(status_code=409, detail="该参赛者已退赛，不能重复退赛")

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

    withdrawal = ParticipantWithdrawal(
        tournament_id=tournament_id,
        participant_id=participant.id,
        request_id=req.request_id,
        expected_event_seq=req.expected_event_seq,
        reason=req.reason,
        params=_fingerprint(req),
        response={},  # 先占位，响应组装后回填（同事务）
    )
    db.add(withdrawal)
    db.flush()  # 取得 withdrawal.id，供事件与响应使用

    event = append_event(
        db,
        tournament_id,
        PARTICIPANT_WITHDRAWN,
        {
            "withdrawal_id": withdrawal.id,
            "participant_id": participant.id,
            "number": participant.number,
            "name": participant.name,
            "reason": req.reason,
            "expected_event_seq": req.expected_event_seq,
        },
    )

    response = WithdrawalResponse(
        id=withdrawal.id,
        tournament_id=tournament_id,
        participant_id=participant.id,
        number=participant.number,
        name=participant.name,
        reason=req.reason,
        request_id=req.request_id,
        expected_event_seq=req.expected_event_seq,
        event_seq=event.seq,
        withdrawn_at=withdrawal.created_at,
    )
    withdrawal.response = response.model_dump(mode="json")
    db.flush()
    return response, event
