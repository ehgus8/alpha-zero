document.addEventListener("DOMContentLoaded", () => {
  const boardElement = document.getElementById("board");
  const statusElement = document.getElementById("status");
  const playBlackBtn = document.getElementById("play-black-btn");
  const playWhiteBtn = document.getElementById("play-white-btn");

  let gameActive = false;

  const createBoard = (rows, cols) => {
    boardElement.innerHTML = "";
    boardElement.style.gridTemplateColumns = `repeat(${cols}, 40px)`;
    boardElement.style.gridTemplateRows = `repeat(${rows}, 40px)`;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cell = document.createElement("div");
        cell.classList.add("cell");
        cell.dataset.row = r;
        cell.dataset.col = c;
        cell.addEventListener("click", handleCellClick);
        boardElement.appendChild(cell);
      }
    }
  };

  const updateBoard = (board) => {
    const cells = document.querySelectorAll(".cell");
    cells.forEach((cell) => {
      const r = parseInt(cell.dataset.row);
      const c = parseInt(cell.dataset.col);

      // Clear previous stone
      cell.innerHTML = "";

      if (board[0][r][c] === 1) {
        // Player 1 (Black)
        const stone = document.createElement("div");
        stone.classList.add("stone", "black");
        cell.appendChild(stone);
      } else if (board[1][r][c] === 1) {
        // Player 2 (White)
        const stone = document.createElement("div");
        stone.classList.add("stone", "white");
        cell.appendChild(stone);
      }
    });
  };

  const handleCellClick = async (event) => {
    if (!gameActive) return;

    const cell = event.currentTarget;
    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);

    statusElement.textContent = "AI is thinking...";

    try {
      const response = await fetch("/move", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ row, col }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Invalid move");
      }

      const data = await response.json();
      updateBoard(data.board);

      if (data.winner !== -1) {
        gameActive = false;
        if (data.winner === 0) {
          statusElement.textContent = "You win!";
        } else {
          statusElement.textContent = "AI wins!";
        }
      } else {
        statusElement.textContent = "Your turn";
      }
    } catch (error) {
      statusElement.textContent = `Error: ${error.message}`;
      console.error("Error:", error);
    }
  };

  const startNewGame = async (playerColor) => {
    gameActive = true;
    statusElement.textContent = `New game started. Your turn (${playerColor}).`;

    try {
      const response = await fetch("/new_game", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ player_color: playerColor }),
      });
      const data = await response.json();
      createBoard(data.rows, data.cols);
      updateBoard(data.board);
      if (playerColor === "white") {
        statusElement.textContent = "You are White. Your turn.";
      } else {
        statusElement.textContent = "You are Black. Your turn.";
      }
    } catch (error) {
      statusElement.textContent = "Failed to start a new game.";
      console.error("Error:", error);
    }
  };

  playBlackBtn.addEventListener("click", () => startNewGame("black"));
  playWhiteBtn.addEventListener("click", () => startNewGame("white"));
});
