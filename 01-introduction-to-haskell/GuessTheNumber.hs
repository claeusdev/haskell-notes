-- GuessTheNumber.hs
-- A number guessing game to demonstrate basic Haskell concepts
-- This project covers: IO, recursion, pattern matching, random numbers

import System.Random
import Control.Monad (when)

-- Data type to represent game state
data GameState = GameState
    { secret :: Int
    , attempts :: Int
    , maxAttempts :: Int
    } deriving Show

-- Function to create a new game state
newGame :: Int -> Int -> GameState
newGame secretNumber maxAttempts = GameState secretNumber 0 maxAttempts

-- Function to increment attempts
incrementAttempts :: GameState -> GameState
incrementAttempts game = game { attempts = attempts game + 1 }

-- Function to check if game is over (too many attempts)
isGameOver :: GameState -> Bool
isGameOver game = attempts game >= maxAttempts game

-- Function to provide hints based on the guess
giveHint :: Int -> Int -> String
giveHint guess secret
    | guess < secret = "Too low! Try a higher number."
    | guess > secret = "Too high! Try a lower number."
    | otherwise = "Congratulations! You guessed it!"

-- Function to get a valid number from user input
getValidNumber :: IO Int
getValidNumber = do
    input <- getLine
    case reads input of
        [(number, "")] -> return number
        _ -> do
            putStrLn "Please enter a valid number:"
            getValidNumber

-- Main game loop
gameLoop :: GameState -> IO ()
gameLoop game = do
    -- Check if game is over due to max attempts
    if isGameOver game
        then do
            putStrLn $ "Game over! You've used all " ++ show (maxAttempts game) ++ " attempts."
            putStrLn $ "The secret number was: " ++ show (secret game)
        else do
            -- Display current attempt
            putStrLn $ "\nAttempt " ++ show (attempts game + 1) ++ " of " ++ show (maxAttempts game)
            putStrLn "Enter your guess:"
            
            -- Get user's guess
            guess <- getValidNumber
            
            -- Update game state
            let newGameState = incrementAttempts game
            
            -- Check the guess and provide feedback
            let hint = giveHint guess (secret game)
            putStrLn hint
            
            -- Continue game if not guessed correctly and attempts remaining
            if guess == secret game
                then do
                    putStrLn $ "You won in " ++ show (attempts newGameState) ++ " attempts!"
                    askPlayAgain
                else gameLoop newGameState

-- Function to ask if player wants to play again
askPlayAgain :: IO ()
askPlayAgain = do
    putStrLn "\nWould you like to play again? (y/n):"
    response <- getLine
    case map toLower (take 1 response) of
        "y" -> startNewGame
        "n" -> putStrLn "Thanks for playing!"
        _ -> do
            putStrLn "Please enter 'y' for yes or 'n' for no."
            askPlayAgain
  where
    toLower c
        | c >= 'A' && c <= 'Z' = toEnum (fromEnum c + 32)
        | otherwise = c

-- Function to get difficulty level
getDifficulty :: IO (Int, Int)
getDifficulty = do
    putStrLn "\nChoose difficulty level:"
    putStrLn "1. Easy (1-50, 10 attempts)"
    putStrLn "2. Medium (1-100, 7 attempts)"
    putStrLn "3. Hard (1-200, 5 attempts)"
    putStrLn "Enter your choice (1-3):"
    
    choice <- getValidNumber
    case choice of
        1 -> return (50, 10)
        2 -> return (100, 7)
        3 -> return (200, 5)
        _ -> do
            putStrLn "Invalid choice. Please select 1, 2, or 3."
            getDifficulty

-- Function to start a new game
startNewGame :: IO ()
startNewGame = do
    (maxNum, maxAttempts) <- getDifficulty
    secretNumber <- randomRIO (1, maxNum)
    
    putStrLn $ "\nI'm thinking of a number between 1 and " ++ show maxNum ++ "."
    putStrLn $ "You have " ++ show maxAttempts ++ " attempts to guess it!"
    
    let game = newGame secretNumber maxAttempts
    gameLoop game

-- Display welcome message and game instructions
welcomeMessage :: IO ()
welcomeMessage = do
    putStrLn "======================================="
    putStrLn "       Welcome to Guess the Number!"
    putStrLn "======================================="
    putStrLn ""
    putStrLn "Instructions:"
    putStrLn "- I will think of a secret number"
    putStrLn "- You try to guess it"
    putStrLn "- I'll tell you if your guess is too high or too low"
    putStrLn "- Try to guess it in as few attempts as possible!"
    putStrLn ""

-- Main function
main :: IO ()
main = do
    welcomeMessage
    startNewGame

-- Additional utility functions for extended features

-- Function to show game statistics
showStats :: [Int] -> IO ()
showStats attempts = do
    let totalGames = length attempts
    let averageAttempts = fromIntegral (sum attempts) / fromIntegral totalGames
    putStrLn $ "\nGame Statistics:"
    putStrLn $ "Games played: " ++ show totalGames
    putStrLn $ "Average attempts: " ++ show (round averageAttempts)
    putStrLn $ "Best game: " ++ show (minimum attempts) ++ " attempts"
    putStrLn $ "Worst game: " ++ show (maximum attempts) ++ " attempts"

-- Enhanced version with statistics tracking
-- (This would require modifying the main game flow to track stats)

-- Example of how to compile and run:
-- $ ghc GuessTheNumber.hs
-- $ ./GuessTheNumber

-- Or run directly with:
-- $ runhaskell GuessTheNumber.hs