# Algebraic Data Types - Examples

## Sum Types (Union Types)

### Basic Sum Types
```haskell
-- basic_sum_types.hs

-- Simple enumeration
data Color = Red | Green | Blue | Yellow
    deriving (Show, Eq)

-- Optional values
data Maybe' a = Nothing' | Just' a
    deriving (Show, Eq)

-- Error handling
data Result a b = Success a | Error b
    deriving (Show, Eq)

-- Pattern matching on sum types
colorToRGB :: Color -> (Int, Int, Int)
colorToRGB Red = (255, 0, 0)
colorToRGB Green = (0, 255, 0)
colorToRGB Blue = (0, 0, 255)
colorToRGB Yellow = (255, 255, 0)

processResult :: Result Int String -> String
processResult (Success n) = "Got value: " ++ show n
processResult (Error err) = "Error occurred: " ++ err

main :: IO ()
main = do
    print (colorToRGB Red)              -- (255,0,0)
    print (colorToRGB Blue)             -- (0,0,255)
    putStrLn (processResult (Success 42))   -- "Got value: 42"
    putStrLn (processResult (Error "oops")) -- "Error occurred: oops"
```

### More Complex Sum Types
```haskell
-- complex_sum_types.hs

-- Mathematical expressions
data Expr = Lit Int | Add Expr Expr | Mul Expr Expr | Div Expr Expr
    deriving (Show, Eq)

-- Shapes with different properties
data Shape 
    = Circle Double                    -- radius
    | Rectangle Double Double          -- width, height
    | Triangle Double Double Double    -- three sides
    deriving (Show, Eq)

-- Network messages
data Message 
    = Ping
    | Pong
    | Data String
    | Request Int String              -- id, query
    | Response Int String             -- id, result
    deriving (Show, Eq)

-- Expression evaluation
evaluate :: Expr -> Either String Int
evaluate (Lit n) = Right n
evaluate (Add e1 e2) = do
    v1 <- evaluate e1
    v2 <- evaluate e2
    Right (v1 + v2)
evaluate (Mul e1 e2) = do
    v1 <- evaluate e1
    v2 <- evaluate e2
    Right (v1 * v2)
evaluate (Div e1 e2) = do
    v1 <- evaluate e1
    v2 <- evaluate e2
    if v2 == 0 
        then Left "Division by zero"
        else Right (v1 `div` v2)

-- Shape calculations
area :: Shape -> Double
area (Circle r) = pi * r * r
area (Rectangle w h) = w * h
area (Triangle a b c) = 
    let s = (a + b + c) / 2
    in sqrt (s * (s - a) * (s - b) * (s - c))

perimeter :: Shape -> Double
perimeter (Circle r) = 2 * pi * r
perimeter (Rectangle w h) = 2 * (w + h)
perimeter (Triangle a b c) = a + b + c

-- Message processing
processMessage :: Message -> String
processMessage Ping = "Received ping"
processMessage Pong = "Received pong"
processMessage (Data content) = "Data: " ++ content
processMessage (Request id query) = "Request " ++ show id ++ ": " ++ query
processMessage (Response id result) = "Response " ++ show id ++ ": " ++ result

main :: IO ()
main = do
    let expr = Add (Mul (Lit 3) (Lit 4)) (Lit 5)
    print (evaluate expr)               -- Right 17
    
    let circle = Circle 5
    let rect = Rectangle 4 6
    print (area circle)                 -- 78.53981633974483
    print (area rect)                   -- 24.0
    print (perimeter circle)            -- 31.41592653589793
    
    putStrLn (processMessage Ping)      -- "Received ping"
    putStrLn (processMessage (Data "hello"))  -- "Data: hello"
```

## Product Types

### Records and Tuples
```haskell
-- product_types.hs

-- Simple product type as tuple
type Point2D = (Double, Double)
type Point3D = (Double, Double, Double)

-- Product type as record
data Person = Person
    { firstName :: String
    , lastName :: String
    , age :: Int
    , email :: String
    } deriving (Show, Eq)

-- Complex record with multiple fields
data BankAccount = BankAccount
    { accountNumber :: String
    , balance :: Double
    , accountType :: AccountType
    , owner :: Person
    , isActive :: Bool
    } deriving (Show, Eq)

data AccountType = Checking | Savings | Investment
    deriving (Show, Eq)

-- Tuple operations
distance2D :: Point2D -> Point2D -> Double
distance2D (x1, y1) (x2, y2) = sqrt ((x2 - x1)^2 + (y2 - y1)^2)

midpoint2D :: Point2D -> Point2D -> Point2D
midpoint2D (x1, y1) (x2, y2) = ((x1 + x2) / 2, (y1 + y2) / 2)

-- Record operations
fullName :: Person -> String
fullName person = firstName person ++ " " ++ lastName person

birthday :: Person -> Person
birthday person = person { age = age person + 1 }

updateEmail :: String -> Person -> Person
updateEmail newEmail person = person { email = newEmail }

-- Bank account operations
deposit :: Double -> BankAccount -> BankAccount
deposit amount account = account { balance = balance account + amount }

withdraw :: Double -> BankAccount -> Either String BankAccount
withdraw amount account
    | not (isActive account) = Left "Account is inactive"
    | amount > balance account = Left "Insufficient funds"
    | amount <= 0 = Left "Invalid amount"
    | otherwise = Right account { balance = balance account - amount }

accountSummary :: BankAccount -> String
accountSummary account = unlines
    [ "Account: " ++ accountNumber account
    , "Owner: " ++ fullName (owner account)
    , "Type: " ++ show (accountType account)
    , "Balance: $" ++ show (balance account)
    , "Status: " ++ if isActive account then "Active" else "Inactive"
    ]

main :: IO ()
main = do
    let p1 = (1.0, 2.0)
    let p2 = (4.0, 6.0)
    print (distance2D p1 p2)            -- 5.0
    print (midpoint2D p1 p2)            -- (2.5,4.0)
    
    let person = Person "John" "Doe" 30 "john@example.com"
    putStrLn (fullName person)          -- "John Doe"
    
    let olderPerson = birthday person
    print (age olderPerson)             -- 31
    
    let account = BankAccount "12345" 1000.0 Checking person True
    putStrLn (accountSummary account)
    
    case withdraw 500 account of
        Left err -> putStrLn err
        Right newAccount -> putStrLn $ "New balance: " ++ show (balance newAccount)
```

## Recursive Data Types

### Lists
```haskell
-- recursive_lists.hs

-- Custom list implementation
data List a = Nil | Cons a (List a)
    deriving (Show, Eq)

-- Convert between built-in and custom lists
fromBuiltinList :: [a] -> List a
fromBuiltinList [] = Nil
fromBuiltinList (x:xs) = Cons x (fromBuiltinList xs)

toBuiltinList :: List a -> [a]
toBuiltinList Nil = []
toBuiltinList (Cons x xs) = x : toBuiltinList xs

-- List operations
lengthList :: List a -> Int
lengthList Nil = 0
lengthList (Cons _ xs) = 1 + lengthList xs

appendList :: List a -> List a -> List a
appendList Nil ys = ys
appendList (Cons x xs) ys = Cons x (appendList xs ys)

reverseList :: List a -> List a
reverseList list = reverseHelper list Nil
  where
    reverseHelper Nil acc = acc
    reverseHelper (Cons x xs) acc = reverseHelper xs (Cons x acc)

mapList :: (a -> b) -> List a -> List b
mapList _ Nil = Nil
mapList f (Cons x xs) = Cons (f x) (mapList f xs)

filterList :: (a -> Bool) -> List a -> List a
filterList _ Nil = Nil
filterList p (Cons x xs)
    | p x = Cons x (filterList p xs)
    | otherwise = filterList p xs

foldList :: (a -> b -> b) -> b -> List a -> b
foldList _ z Nil = z
foldList f z (Cons x xs) = f x (foldList f z xs)

main :: IO ()
main = do
    let myList = fromBuiltinList [1,2,3,4,5]
    print myList                        -- Cons 1 (Cons 2 (Cons 3 (Cons 4 (Cons 5 Nil))))
    print (lengthList myList)           -- 5
    print (toBuiltinList (reverseList myList))  -- [5,4,3,2,1]
    print (toBuiltinList (mapList (*2) myList)) -- [2,4,6,8,10]
    print (toBuiltinList (filterList even myList)) -- [2,4]
    print (foldList (+) 0 myList)       -- 15
```

### Binary Trees
```haskell
-- binary_trees.hs

-- Binary tree definition
data Tree a = Empty | Node a (Tree a) (Tree a)
    deriving (Show, Eq)

-- Tree construction
singleton :: a -> Tree a
singleton x = Node x Empty Empty

insert :: Ord a => a -> Tree a -> Tree a
insert x Empty = singleton x
insert x (Node y left right)
    | x <= y = Node y (insert x left) right
    | otherwise = Node y left (insert x right)

-- Tree traversals
inorder :: Tree a -> [a]
inorder Empty = []
inorder (Node x left right) = inorder left ++ [x] ++ inorder right

preorder :: Tree a -> [a]
preorder Empty = []
preorder (Node x left right) = [x] ++ preorder left ++ preorder right

postorder :: Tree a -> [a]
postorder Empty = []
postorder (Node x left right) = postorder left ++ postorder right ++ [x]

-- Tree properties
treeSize :: Tree a -> Int
treeSize Empty = 0
treeSize (Node _ left right) = 1 + treeSize left + treeSize right

treeHeight :: Tree a -> Int
treeHeight Empty = 0
treeHeight (Node _ left right) = 1 + max (treeHeight left) (treeHeight right)

treeSum :: Num a => Tree a -> a
treeSum Empty = 0
treeSum (Node x left right) = x + treeSum left + treeSum right

-- Tree operations
treeMap :: (a -> b) -> Tree a -> Tree b
treeMap _ Empty = Empty
treeMap f (Node x left right) = Node (f x) (treeMap f left) (treeMap f right)

treeFold :: (a -> b -> b -> b) -> b -> Tree a -> b
treeFold _ z Empty = z
treeFold f z (Node x left right) = f x (treeFold f z left) (treeFold f z right)

-- Search operations
treeSearch :: Ord a => a -> Tree a -> Bool
treeSearch _ Empty = False
treeSearch x (Node y left right)
    | x == y = True
    | x < y = treeSearch x left
    | otherwise = treeSearch x right

treeMin :: Tree a -> Maybe a
treeMin Empty = Nothing
treeMin (Node x Empty _) = Just x
treeMin (Node _ left _) = treeMin left

treeMax :: Tree a -> Maybe a
treeMax Empty = Nothing
treeMax (Node x _ Empty) = Just x
treeMax (Node _ _ right) = treeMax right

-- Build tree from list
fromList :: Ord a => [a] -> Tree a
fromList = foldr insert Empty

main :: IO ()
main = do
    let tree = fromList [5, 3, 7, 1, 4, 6, 8]
    print tree
    print (inorder tree)                -- [1,3,4,5,6,7,8]
    print (preorder tree)               -- [5,3,1,4,7,6,8]
    print (postorder tree)              -- [1,4,3,6,8,7,5]
    print (treeSize tree)               -- 7
    print (treeHeight tree)             -- 3
    print (treeSum tree)                -- 34
    print (treeSearch 4 tree)           -- True
    print (treeSearch 9 tree)           -- False
    print (treeMin tree)                -- Just 1
    print (treeMax tree)                -- Just 8
```

## Advanced ADT Patterns

### Phantom Types
```haskell
-- phantom_types.hs

-- Temperature with phantom type parameter
data Temperature unit = Temperature Double
    deriving (Show, Eq)

-- Units as phantom types
data Celsius
data Fahrenheit
data Kelvin

-- Type-safe temperature conversions
celsiusToFahrenheit :: Temperature Celsius -> Temperature Fahrenheit
celsiusToFahrenheit (Temperature c) = Temperature (c * 9/5 + 32)

fahrenheitToCelsius :: Temperature Fahrenheit -> Temperature Celsius
fahrenheitToCelsius (Temperature f) = Temperature ((f - 32) * 5/9)

celsiusToKelvin :: Temperature Celsius -> Temperature Kelvin
celsiusToKelvin (Temperature c) = Temperature (c + 273.15)

kelvinToCelsius :: Temperature Kelvin -> Temperature Celsius
kelvinToCelsius (Temperature k) = Temperature (k - 273.15)

-- Distance with units
data Distance unit = Distance Double
    deriving (Show, Eq)

data Meters
data Feet
data Miles

metersToFeet :: Distance Meters -> Distance Feet
metersToFeet (Distance m) = Distance (m * 3.28084)

feetToMeters :: Distance Feet -> Distance Meters
feetToMeters (Distance f) = Distance (f / 3.28084)

-- Type-safe calculations
addTemperatures :: Temperature unit -> Temperature unit -> Temperature unit
addTemperatures (Temperature a) (Temperature b) = Temperature (a + b)

-- This won't compile - mixing units
-- invalidOperation = addTemperatures (Temperature 100 :: Temperature Celsius) 
--                                   (Temperature 32 :: Temperature Fahrenheit)

main :: IO ()
main = do
    let tempC = Temperature 25 :: Temperature Celsius
    let tempF = celsiusToFahrenheit tempC
    let tempK = celsiusToKelvin tempC
    
    print tempC                         -- Temperature 25.0
    print tempF                         -- Temperature 77.0
    print tempK                         -- Temperature 298.15
    
    let distM = Distance 100 :: Distance Meters
    let distF = metersToFeet distM
    print distM                         -- Distance 100.0
    print distF                         -- Distance 328.084
```

### Newtype Wrappers
```haskell
-- newtype_wrappers.hs

-- Type-safe identifiers
newtype UserId = UserId Int
    deriving (Show, Eq, Ord)

newtype ProductId = ProductId String
    deriving (Show, Eq, Ord)

newtype OrderId = OrderId String
    deriving (Show, Eq, Ord)

-- Type-safe quantities
newtype Quantity = Quantity Int
    deriving (Show, Eq, Ord, Num)

newtype Price = Price Double
    deriving (Show, Eq, Ord, Num)

-- Business domain types
data User = User
    { userId :: UserId
    , userName :: String
    , userEmail :: String
    } deriving (Show, Eq)

data Product = Product
    { productId :: ProductId
    , productName :: String
    , productPrice :: Price
    } deriving (Show, Eq)

data OrderItem = OrderItem
    { itemProduct :: ProductId
    , itemQuantity :: Quantity
    , itemPrice :: Price
    } deriving (Show, Eq)

data Order = Order
    { orderId :: OrderId
    , orderUser :: UserId
    , orderItems :: [OrderItem]
    } deriving (Show, Eq)

-- Type-safe operations
calculateItemTotal :: OrderItem -> Price
calculateItemTotal (OrderItem _ (Quantity qty) (Price price)) = Price (fromIntegral qty * price)

calculateOrderTotal :: Order -> Price
calculateOrderTotal order = sum (map calculateItemTotal (orderItems order))

-- Helper functions with type safety
createUser :: Int -> String -> String -> User
createUser id name email = User (UserId id) name email

createProduct :: String -> String -> Double -> Product
createProduct id name price = Product (ProductId id) name (Price price)

addItemToOrder :: OrderItem -> Order -> Order
addItemToOrder item order = order { orderItems = item : orderItems order }

-- This prevents mixing up different ID types:
-- getUserProduct :: UserId -> Product  -- This would be caught at compile time
-- getUserProduct (ProductId "123") = undefined

main :: IO ()
main = do
    let user = createUser 1 "John Doe" "john@example.com"
    let product = createProduct "PROD001" "Widget" 29.99
    
    let item = OrderItem (ProductId "PROD001") (Quantity 3) (Price 29.99)
    let order = Order (OrderId "ORD001") (UserId 1) [item]
    
    print user
    print product
    print order
    print (calculateItemTotal item)     -- Price 89.97
    print (calculateOrderTotal order)   -- Price 89.97
```

### GADTs (Generalized Algebraic Data Types)
```haskell
-- gadts.hs
{-# LANGUAGE GADTs #-}

-- Expression GADT with type safety
data Expr a where
    LitInt    :: Int -> Expr Int
    LitBool   :: Bool -> Expr Bool
    LitString :: String -> Expr String
    Add       :: Expr Int -> Expr Int -> Expr Int
    Mult      :: Expr Int -> Expr Int -> Expr Int
    Equal     :: Eq a => Expr a -> Expr a -> Expr Bool
    IfThenElse :: Expr Bool -> Expr a -> Expr a -> Expr a

-- Type-safe evaluation
eval :: Expr a -> a
eval (LitInt n) = n
eval (LitBool b) = b
eval (LitString s) = s
eval (Add e1 e2) = eval e1 + eval e2
eval (Mult e1 e2) = eval e1 * eval e2
eval (Equal e1 e2) = eval e1 == eval e2
eval (IfThenElse cond thenExpr elseExpr) = 
    if eval cond then eval thenExpr else eval elseExpr

-- Vector with length in type
data Vec n a where
    VNil  :: Vec 0 a
    VCons :: a -> Vec n a -> Vec (n + 1) a

-- Type-safe vector operations
vhead :: Vec (n + 1) a -> a
vhead (VCons x _) = x

vtail :: Vec (n + 1) a -> Vec n a
vtail (VCons _ xs) = xs

vappend :: Vec n a -> Vec m a -> Vec (n + m) a
vappend VNil ys = ys
vappend (VCons x xs) ys = VCons x (vappend xs ys)

main :: IO ()
main = do
    let expr1 = Add (LitInt 5) (LitInt 3)
    let expr2 = IfThenElse (Equal (LitInt 2) (LitInt 2)) (LitString "equal") (LitString "not equal")
    
    print (eval expr1)                  -- 8
    putStrLn (eval expr2)               -- "equal"
    
    let vec1 = VCons 1 (VCons 2 (VCons 3 VNil))
    let vec2 = VCons 4 (VCons 5 VNil)
    let combined = vappend vec1 vec2
    
    print (vhead vec1)                  -- 1
    print (vhead (vtail vec1))          -- 2
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

These examples demonstrate how Algebraic Data Types enable precise modeling of your problem domain while maintaining type safety and expressiveness.