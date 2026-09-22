import chess
import pytest

from chess_ai.players import GameContext, RandomPlayer

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        # In check, with Kxg2 the only legal move.
        "7k/8/8/8/8/8/6q1/7K w - - 0 1",
        # Castling, en passant and promotions available.
        "r3k2r/1P6/8/3pP3/8/8/8/R3K2R w KQkq d6 0 2",
    ],
)
async def test_random_player_chooses_a_legal_move(fen):
    board = chess.Board(fen)
    player = RandomPlayer(seed=0)

    for _ in range(20):
        choice = await player.choose_move(GameContext(board.copy()))

        assert choice.move in board.legal_moves
        assert choice.thoughts is None


async def test_random_player_is_reproducible_with_a_seed():
    board = chess.Board()

    async def moves(player: RandomPlayer) -> list[chess.Move]:
        return [(await player.choose_move(GameContext(board.copy()))).move for _ in range(20)]

    assert await moves(RandomPlayer(seed=7)) == await moves(RandomPlayer(seed=7))
    assert len(set(await moves(RandomPlayer(seed=7)))) > 1
