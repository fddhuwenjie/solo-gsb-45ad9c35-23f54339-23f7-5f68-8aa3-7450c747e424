"""选手永久退赛接口。

- POST /tournaments/{id}/withdrawals：为参赛者办理永久退赛。请求携带
  幂等键 request_id、乐观快照 expected_event_seq、参赛者 id 与原因；
  仅在当前没有未结束轮次、待处理复核、未消费的下一轮签到快照或已发布
  晋级名单时生效。成功时在同一写事务内写入 participant.withdrawn 事件。
- GET /tournaments/{id}/withdrawals：赛事下的全部退赛记录（按编号）。
- GET /tournaments/{id}/participants/{pid}/withdrawal：某参赛者的退赛
  记录（未退赛 404）。

重复 request_id（参数一致）原样返回首次结果，不产生新写入与事件；同键
不同参数、序号冲突、重复退赛一律 409 且不落库。
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..events.service import commit_and_publish
from ..locking import writer_lock
from ..models import Participant
from ..schemas import WithdrawalRequest, WithdrawalResponse
from ..services.withdrawals import (
    get_withdrawal,
    list_withdrawals,
    serialize_withdrawal,
    withdraw_participant,
)
from .tournaments import get_tournament_or_404

router = APIRouter()


@router.post(
    "/tournaments/{tournament_id}/withdrawals",
    response_model=WithdrawalResponse,
    status_code=201,
)
def withdraw(
    tournament_id: int,
    payload: WithdrawalRequest,
    db: Session = Depends(get_db),
):
    get_tournament_or_404(db, tournament_id)
    # 写请求进程级串行：资格校验、快照确认、退赛落库、事件序号与
    # 「提交→发布」全程持锁
    with writer_lock():
        db.expire_all()
        response, event = withdraw_participant(db, tournament_id, payload)
        # 幂等重放（event 为 None）没有任何写入，直接返回首次结果
        if event is not None:
            commit_and_publish(db, [event])
    return response


@router.get(
    "/tournaments/{tournament_id}/withdrawals",
    response_model=list[WithdrawalResponse],
)
def list_tournament_withdrawals(
    tournament_id: int, db: Session = Depends(get_db)
):
    get_tournament_or_404(db, tournament_id)
    return [
        serialize_withdrawal(w, w.participant)
        for w in list_withdrawals(db, tournament_id)
    ]


@router.get(
    "/tournaments/{tournament_id}/participants/{participant_id}/withdrawal",
    response_model=WithdrawalResponse,
)
def get_participant_withdrawal(
    tournament_id: int,
    participant_id: int,
    db: Session = Depends(get_db),
):
    get_tournament_or_404(db, tournament_id)
    participant = (
        db.query(Participant)
        .filter(
            Participant.id == participant_id,
            Participant.tournament_id == tournament_id,
        )
        .first()
    )
    if participant is None:
        raise HTTPException(status_code=404, detail="参赛者不存在")
    withdrawal = get_withdrawal(db, tournament_id, participant_id)
    if withdrawal is None:
        raise HTTPException(status_code=404, detail="该参赛者未退赛")
    return serialize_withdrawal(withdrawal, participant)
