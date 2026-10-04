"""选手永久退赛接口。

- POST /tournaments/{id}/withdrawals：为参赛者办理永久退赛。请求携带
  幂等键 request_id、乐观快照 expected_event_seq、参赛者 participant_id
  与退赛理由；仅在没有未结束轮次、待处理赛果复核、未消费的下一轮签到
  快照或已发布晋级名单时生效。退赛状态与 participant.withdrawn 事件在
  同一写事务内提交。
- GET /tournaments/{id}/withdrawals：列出赛事全部退赛记录（按退赛顺序）。

重复 request_id（参数一致）原样返回首次结果，不产生新写入与事件；
同键不同参数、序号冲突或重复退赛一律 409 且不落库。退赛保留历史
对局、积分与扣分；排名查询保留其历史数据并标记 withdrawn。
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..events.service import commit_and_publish
from ..locking import writer_lock
from ..models import WithdrawalAction
from ..schemas import WithdrawalRequest, WithdrawalResponse
from ..services.withdrawals import withdraw_participant
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
    # 写请求进程级串行：资格校验、快照确认、退赛状态落库、事件序号与
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
def list_withdrawals(tournament_id: int, db: Session = Depends(get_db)):
    get_tournament_or_404(db, tournament_id)
    records = (
        db.query(WithdrawalAction)
        .filter(WithdrawalAction.tournament_id == tournament_id)
        .order_by(WithdrawalAction.id)
        .all()
    )
    return [WithdrawalResponse(**r.response) for r in records]
