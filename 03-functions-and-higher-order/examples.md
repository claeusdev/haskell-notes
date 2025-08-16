# Functions and Higher-Order Functions - Examples

## Function Definition Examples

### Basic Function Definitions
```haskell
-- basic_functions.hs

-- Simple functions
square :: Int -> Int
square x = x * x

cube :: Int -> Int
cube x = x * x * x

-- Function with multiple parameters
power :: Int -> Int -> Int
power base exponent = base ^ exponent

-- Function using guards
absoluteValue :: Int -> Int
absoluteValue n
    | n >= 0 = n
    | otherwise = -n

-- Function with where clause
triangleArea :: Double -> Double -> Double -> Double
triangleArea a b c = sqrt (s * (s - a) * (s - b) * (s - c))
  where s = (a + b + c) / 2

main :: IO ()
main = do
    print (square 5)            -- 25
    print (cube 3)              -- 27
    print (power 2 8)           -- 256
    print (absoluteValue (-10)) -- 10
    print (triangleArea 3 4 5)  -- 6.0
```

### Function Composition Examples
```haskell
-- composition.hs

-- Basic composition
double :: Int -> Int
double x = x * 2

square :: Int -> Int
square x = x * x

-- Composed function
doubleAndSquare :: Int -> Int
doubleAndSquare = square . double

-- Multiple composition
processText :: String -> String
processText = reverse . map toUpper . filter isAlpha

-- Application operator examples
calculate :: Double -> Double
calculate x = sqrt $ (x + 5) * 2

-- Complex composition
analyzeWords :: String -> (Int, Int, Double)
analyzeWords text = (wordCount, charCount, avgLength)
  where
    ws = words text
    wordCount = length ws
    charCount = sum . map length $ ws
    avgLength = if wordCount > 0 
                then fromIntegral charCount / fromIntegral wordCount 
                else 0

main :: IO ()
main = do
    print (doubleAndSquare 5)           -- 100 (5 * 2 = 10, 10^2 = 100)
    putStrLn (processText "Hello123")   -- "OLLEH"
    print (calculate 3)                 -- 4.0
    print (analyzeWords "hello world")  -- (2,10,5.0)
```

## Currying and Partial Application Examples

### Understanding Currying
```haskell
-- currying.hs

-- Basic curried function
add :: Int -> Int -> Int
add x y = x + y

-- Partial applications
addFive :: Int -> Int
addFive = add 5

addTen :: Int -> Int
addTen = add 10

-- Operator sections
increment :: Int -> Int
increment = (+1)

double :: Int -> Int
double = (*2)

halve :: Double -> Double
halve = (/2)

isGreaterThanFive :: Int -> Bool
isGreaterThanFive = (>5)

-- Custom partial application
multiplyBy :: Num a => a -> a -> a
multiplyBy factor x = factor * x

triple :: Num a => a -> a
triple = multiplyBy 3

-- Function factories
makeAdder :: Int -> (Int -> Int)
makeAdder n = \x -> x + n

makeMultiplier :: Int -> (Int -> Int)
makeMultiplier n = \x -> x * n

main :: IO ()
main = do
    print (addFive 3)           -- 8
    print (addTen 7)            -- 17
    print (increment 9)         -- 10
    print (double 6)            -- 12
    print (halve 20)            -- 10.0
    print (isGreaterThanFive 3) -- False
    print (triple 4)            -- 12
    
    let add2 = makeAdder 2
    let mult3 = makeMultiplier 3
    print (add2 5)              -- 7
    print (mult3 4)             -- 12
```

## Higher-Order Function Examples

### Map Examples
```haskell
-- map_examples.hs

-- Basic map usage
squares :: [Int] -> [Int]
squares = map (^2)

lengths :: [String] -> [Int]
lengths = map length

-- Nested map
doubleMatrix :: [[Int]] -> [[Int]]
doubleMatrix = map (map (*2))

-- Map with custom function
celsiusToFahrenheit :: [Double] -> [Double]
celsiusToFahrenheit = map (\c -> c * 9/5 + 32)

-- Map with partial application
addToAll :: Int -> [Int] -> [Int]
addToAll n = map (+n)

-- Complex transformations
processNames :: [String] -> [String]
processNames = map (map toUpper . take 3)

main :: IO ()
main = do
    print (squares [1,2,3,4,5])                    -- [1,4,9,16,25]
    print (lengths ["hello", "world", "!"])        -- [5,5,1]
    print (doubleMatrix [[1,2],[3,4]])             -- [[2,4],[6,8]]
    print (celsiusToFahrenheit [0, 20, 100])       -- [32.0,68.0,212.0]
    print (addToAll 10 [1,2,3])                    -- [11,12,13]
    print (processNames ["alice", "bob", "charlie"]) -- ["ALI","BOB","CHA"]
```

### Filter Examples
```haskell
-- filter_examples.hs

-- Basic filtering
evens :: [Int] -> [Int]
evens = filter even

positives :: [Int] -> [Int]
positives = filter (>0)

-- String filtering
longWords :: [String] -> [String]
longWords = filter ((>5) . length)

vowels :: String -> String
vowels = filter (`elem` "aeiouAEIOU")

-- Complex predicates
primes :: [Int] -> [Int]
primes = filter isPrime
  where
    isPrime n = n > 1 && all (\x -> n `mod` x /= 0) [2..floor $ sqrt $ fromIntegral n]

-- Filtering with custom conditions
validEmails :: [String] -> [String]
validEmails = filter isValidEmail
  where
    isValidEmail email = '@' `elem` email && '.' `elem` email

-- Combining filters
processNumbers :: [Int] -> [Int]
processNumbers = filter even . filter (>10) . filter (<100)

main :: IO ()
main = do
    print (evens [1,2,3,4,5,6])                    -- [2,4,6]
    print (positives [-2,-1,0,1,2])                -- [1,2]
    print (longWords ["hi", "hello", "world"])     -- ["hello","world"]
    putStrLn (vowels "programming")                 -- "oai"
    print (primes [1..20])                         -- [2,3,5,7,11,13,17,19]
    print (validEmails ["user@domain.com", "invalid"]) -- ["user@domain.com"]
    print (processNumbers [5,15,25,35,45,95,105])  -- [96] -- wait, this is wrong
    print (processNumbers [5,15,25,35,45,95,105])  -- []
```

### Fold Examples
```haskell
-- fold_examples.hs
import Data.List (foldl')

-- Basic folds
sumList :: [Int] -> Int
sumList = foldr (+) 0

productList :: [Int] -> Int
productList = foldr (*) 1

-- String operations with fold
concatenateStrings :: [String] -> String
concatenateStrings = foldr (++) ""

reverseList :: [a] -> [a]
reverseList = foldl (flip (:)) []

-- Maximum and minimum
maximumSafe :: Ord a => [a] -> Maybe a
maximumSafe [] = Nothing
maximumSafe (x:xs) = Just $ foldl max x xs

-- Counting with fold
countOccurrences :: Eq a => a -> [a] -> Int
countOccurrences x = foldr (\y acc -> if x == y then acc + 1 else acc) 0

-- Building data structures
listToTuple :: [a] -> (Int, [a])
listToTuple xs = foldr (\x (count, list) -> (count + 1, x:list)) (0, []) xs

-- Complex fold example: word frequency
wordFrequency :: String -> [(String, Int)]
wordFrequency text = foldr addWord [] (words text)
  where
    addWord word [] = [(word, 1)]
    addWord word ((w, count):rest)
        | word == w = (w, count + 1) : rest
        | otherwise = (w, count) : addWord word rest

main :: IO ()
main = do
    print (sumList [1,2,3,4,5])                    -- 15
    print (productList [1,2,3,4])                  -- 24
    putStrLn (concatenateStrings ["Hello", " ", "World"]) -- "Hello World"
    print (reverseList [1,2,3,4])                  -- [4,3,2,1]
    print (maximumSafe [3,1,4,1,5])                -- Just 5
    print (maximumSafe ([] :: [Int]))               -- Nothing
    print (countOccurrences 'l' "hello")           -- 2
    print (listToTuple [1,2,3])                    -- (3,[1,2,3])
    print (wordFrequency "hello world hello")      -- [("hello",2),("world",1)]
```

### ZipWith Examples
```haskell
-- zipwith_examples.hs

-- Basic zipWith operations
addLists :: [Int] -> [Int] -> [Int]
addLists = zipWith (+)

multiplyLists :: [Int] -> [Int] -> [Int]
multiplyLists = zipWith (*)

-- String operations
combineNames :: [String] -> [String] -> [String]
combineNames = zipWith (\first last -> first ++ " " ++ last)

-- Distance calculation
dotProduct :: [Double] -> [Double] -> Double
dotProduct xs ys = sum $ zipWith (*) xs ys

-- Complex zipWith
zipWithIndex :: [a] -> [(Int, a)]
zipWithIndex xs = zipWith (,) [0..] xs

-- Multiple lists
zipWith3Example :: [Int] -> [Int] -> [Int] -> [Int]
zipWith3Example = zipWith3 (\x y z -> x + y + z)

-- Custom zipWith function
zipWithDefault :: a -> b -> (a -> b -> c) -> [a] -> [b] -> [c]
zipWithDefault defA defB f [] ys = map (f defA) ys
zipWithDefault defA defB f xs [] = map (\x -> f x defB) xs
zipWithDefault defA defB f (x:xs) (y:ys) = f x y : zipWithDefault defA defB f xs ys

main :: IO ()
main = do
    print (addLists [1,2,3] [4,5,6])              -- [5,7,9]
    print (multiplyLists [2,3,4] [5,6,7])         -- [10,18,28]
    print (combineNames ["John", "Jane"] ["Doe", "Smith"]) -- ["John Doe","Jane Smith"]
    print (dotProduct [1,2,3] [4,5,6])            -- 32.0
    print (zipWithIndex ['a','b','c'])            -- [(0,'a'),(1,'b'),(2,'c')]
    print (zipWith3Example [1,2] [3,4] [5,6])     -- [9,12]
    print (zipWithDefault 0 0 (+) [1,2,3] [4,5])  -- [5,7,3]
```

## Advanced Higher-Order Examples

### Function Pipelines
```haskell
-- pipelines.hs
import Data.Char (toUpper, isAlpha)
import Data.List (sortBy)

-- Text processing pipeline
processText :: String -> [String]
processText = map reverse        -- 4. Reverse each word
            . sortBy compareLength -- 3. Sort by length
            . filter (not . null)  -- 2. Remove empty strings
            . map (filter isAlpha) -- 1. Keep only letters
            . words                -- 0. Split into words
  where
    compareLength a b = compare (length a) (length b)

-- Data analysis pipeline
analyzeNumbers :: [Int] -> (Double, Double, Int)
analyzeNumbers nums = (mean, stdDev, count)
  where
    count = length positives
    positives = filter (>0) nums
    mean = fromIntegral (sum positives) / fromIntegral count
    variance = sum (map (\x -> (fromIntegral x - mean)^2) positives) / fromIntegral count
    stdDev = sqrt variance

-- File processing pipeline (conceptual)
processLogFile :: String -> [(String, Int)]
processLogFile = map (\ws -> (head ws, length ws))  -- 4. Count occurrences
               . groupBy (==)                        -- 3. Group identical
               . sort                               -- 2. Sort
               . map (takeWhile (/= ' '))           -- 1. Extract first word
               . lines                              -- 0. Split into lines

-- Point-free style examples
-- Calculate sum of squares of even numbers
sumSquareEvens :: [Int] -> Int
sumSquareEvens = sum . map (^2) . filter even

-- Count words in text
wordCount :: String -> Int
wordCount = length . words

-- Get initials from full name
getInitials :: String -> String
getInitials = map head . words

main :: IO ()
main = do
    print (processText "hello world 123 programming!")
    print (analyzeNumbers [1,2,3,4,5,-1,-2])
    print (sumSquareEvens [1,2,3,4,5,6])      -- 56 (4+16+36)
    print (wordCount "Hello beautiful world") -- 3
    putStrLn (getInitials "John Doe Smith")   -- "JDS"
```

### Function Factories and Combinators
```haskell
-- combinators.hs

-- Function combinators
identity :: a -> a
identity x = x

compose :: (b -> c) -> (a -> b) -> (a -> c)
compose f g x = f (g x)

flip' :: (a -> b -> c) -> b -> a -> c
flip' f x y = f y x

-- Function factories
makeValidator :: (a -> Bool) -> String -> (a -> Either String a)
makeValidator predicate errorMsg = \x ->
    if predicate x
        then Right x
        else Left errorMsg

makePredicate :: (a -> a -> Bool) -> a -> (a -> Bool)
makePredicate comparison threshold = \x -> comparison x threshold

-- Specific validators
positiveValidator :: Int -> Either String Int
positiveValidator = makeValidator (>0) "Must be positive"

nonEmptyValidator :: String -> Either String String
nonEmptyValidator = makeValidator (not . null) "Must not be empty"

lengthValidator :: Int -> String -> Either String String
lengthValidator minLen = makeValidator ((>=minLen) . length) 
                                     ("Must be at least " ++ show minLen ++ " characters")

-- Predicate factories
greaterThan :: Ord a => a -> (a -> Bool)
greaterThan = makePredicate (>)

lessThan :: Ord a => a -> (a -> Bool)
lessThan = makePredicate (<)

-- Function modifiers
twice :: (a -> a) -> (a -> a)
twice f x = f (f x)

nTimes :: Int -> (a -> a) -> (a -> a)
nTimes 0 f = identity
nTimes n f = f . nTimes (n-1) f

-- Conditional execution
when' :: Bool -> (a -> a) -> (a -> a)
when' True f = f
when' False _ = identity

unless :: Bool -> (a -> a) -> (a -> a)
unless = when' . not

main :: IO ()
main = do
    print (positiveValidator 5)              -- Right 5
    print (positiveValidator (-3))           -- Left "Must be positive"
    print (nonEmptyValidator "hello")        -- Right "hello"
    print (nonEmptyValidator "")             -- Left "Must not be empty"
    
    let gt5 = greaterThan 5
    let lt10 = lessThan 10
    print (gt5 7)                            -- True
    print (lt10 15)                          -- False
    
    let quadruple = twice (*2)
    print (quadruple 3)                      -- 12
    
    let add1FiveTimes = nTimes 5 (+1)
    print (add1FiveTimes 0)                  -- 5
```

### Recursive Higher-Order Functions
```haskell
-- recursive_higher_order.hs

-- Binary tree data type
data Tree a = Empty | Node a (Tree a) (Tree a)
    deriving (Show, Eq)

-- Higher-order functions on trees
mapTree :: (a -> b) -> Tree a -> Tree b
mapTree f Empty = Empty
mapTree f (Node x left right) = Node (f x) (mapTree f left) (mapTree f right)

filterTree :: (a -> Bool) -> Tree a -> Tree a
filterTree p Empty = Empty
filterTree p (Node x left right)
    | p x = Node x (filterTree p left) (filterTree p right)
    | otherwise = Empty  -- Remove node that doesn't satisfy predicate

foldTree :: (a -> b -> b -> b) -> b -> Tree a -> b
foldTree f z Empty = z
foldTree f z (Node x left right) = f x (foldTree f z left) (foldTree f z right)

-- Tree construction
sampleTree :: Tree Int
sampleTree = Node 5 
                (Node 3 (Node 1 Empty Empty) (Node 4 Empty Empty))
                (Node 8 (Node 6 Empty Empty) (Node 9 Empty Empty))

-- List operations with custom fold
foldList :: (a -> b -> b) -> b -> [a] -> b
foldList f z [] = z
foldList f z (x:xs) = f x (foldList f z xs)

-- Custom map using fold
mapWithFold :: (a -> b) -> [a] -> [b]
mapWithFold f = foldList (\x acc -> f x : acc) []

-- Custom filter using fold
filterWithFold :: (a -> Bool) -> [a] -> [a]
filterWithFold p = foldList (\x acc -> if p x then x : acc else acc) []

-- Higher-order function that takes multiple functions
applyAll :: [a -> a] -> a -> a
applyAll fs x = foldList (.) identity fs $ x

-- Function composition chain
composeAll :: [a -> a] -> (a -> a)
composeAll = foldList (.) identity

main :: IO ()
main = do
    print (mapTree (*2) sampleTree)
    print (filterTree even sampleTree)
    print (foldTree (\x l r -> x + l + r) 0 sampleTree)  -- Sum all nodes
    
    print (mapWithFold (*3) [1,2,3,4])      -- [3,6,9,12]
    print (filterWithFold even [1,2,3,4,5]) -- [2,4]
    
    let functions = [(*2), (+1), (*3)]
    print (applyAll functions 2)             -- ((2*2)+1)*3 = 15
    
    let composed = composeAll [(*2), (+1), (*3)]
    print (composed 2)                       -- Same as above: 15
```

## Lambda Function Examples

### Basic Lambda Usage
```haskell
-- lambdas.hs

-- Simple lambda functions
square :: Int -> Int
square = \x -> x * x

add :: Int -> Int -> Int
add = \x y -> x + y

-- Lambda in higher-order functions
numbers :: [Int]
numbers = [1,2,3,4,5]

squared :: [Int]
squared = map (\x -> x * x) numbers

evens :: [Int]
evens = filter (\x -> x `mod` 2 == 0) numbers

-- Lambda with pattern matching
processEither :: [Either String Int] -> [String]
processEither = map (\case
    Left err -> "Error: " ++ err
    Right n -> "Success: " ++ show n)

-- Complex lambda examples
sortByLength :: [String] -> [String]
sortByLength = sortBy (\a b -> compare (length a) (length b))

groupByFirst :: [(Char, Int)] -> [[(Char, Int)]]
groupByFirst = groupBy (\(a,_) (b,_) -> a == b)

-- Lambda returning lambda
makeAdder :: Int -> (Int -> Int)
makeAdder = \n -> \x -> x + n

-- Multi-line lambda
processData :: [String] -> [String]
processData = map (\s -> let trimmed = reverse . dropWhile (== ' ') . reverse $ s
                         in if null trimmed 
                            then "EMPTY"
                            else map toUpper trimmed)

main :: IO ()
main = do
    print (square 5)                         -- 25
    print (add 3 4)                          -- 7
    print squared                            -- [1,4,9,16,25]
    print evens                              -- [2,4]
    
    let eithers = [Left "error", Right 42, Left "oops", Right 7]
    print (processEither eithers)
    
    print (sortByLength ["hello", "hi", "programming", "a"])
    
    let adder = makeAdder 10
    print (adder 5)                          -- 15
```

## Performance and Optimization Examples

### Strict vs Lazy Evaluation
```haskell
-- performance.hs
import Data.List (foldl')

-- Lazy fold (can cause space leaks)
lazySumSquares :: [Int] -> Int
lazySumSquares = foldl (\acc x -> acc + x * x) 0

-- Strict fold (better performance)
strictSumSquares :: [Int] -> Int
strictSumSquares = foldl' (\acc x -> acc + x * x) 0

-- Fusion optimization example
-- GHC can optimize this into a single loop
efficientPipeline :: [Int] -> [Int]
efficientPipeline = map (*3) . filter even . map (+1)

-- Less efficient: intermediate lists
inefficientPipeline :: [Int] -> [Int]
inefficientPipeline xs = 
    let step1 = map (+1) xs
        step2 = filter even step1
        step3 = map (*3) step2
    in step3

-- Tail-recursive higher-order function
mapTailRec :: (a -> b) -> [a] -> [b]
mapTailRec f xs = go xs []
  where
    go [] acc = reverse acc
    go (y:ys) acc = go ys (f y : acc)

-- Avoiding repeated work
memoizedFib :: Int -> Integer
memoizedFib = (map fib [0..] !!)
  where
    fib 0 = 0
    fib 1 = 1
    fib n = memoizedFib (n-1) + memoizedFib (n-2)

main :: IO ()
main = do
    let bigList = [1..100000]
    
    -- These would show performance differences with proper benchmarking
    print $ take 5 $ efficientPipeline [1..20]
    print $ take 5 $ inefficientPipeline [1..20]
    
    print $ mapTailRec (*2) [1,2,3,4,5]
    
    print $ map memoizedFib [10,15,20]
```

## Running the Examples

To run any of these examples:

1. Save the code to a `.hs` file
2. Compile with: `ghc filename.hs`
3. Run with: `./filename`

Or use GHCi for interactive testing:
```bash
ghci filename.hs
*Main> functionName arguments
```

These examples demonstrate the power and expressiveness of higher-order functions in Haskell, showing how they enable elegant, composable, and reusable code.