# Algebraic Data Types - Notes

## Overview

Algebraic Data Types (ADTs) are one of Haskell's most powerful features, allowing you to create precise, type-safe representations of your problem domain. This section covers sum types, product types, recursive types, and advanced ADT patterns.

## Sum Types (Union Types)

### Basic Sum Types
```haskell
data Color = Red | Green | Blue
data Bool = True | False
data Maybe a = Nothing | Just a
data Either a b = Left a | Right b
```

### Pattern Matching on Sum Types
```haskell
colorName :: Color -> String
colorName Red = "red"
colorName Green = "green"
colorName Blue = "blue"

processResult :: Maybe Int -> String
processResult Nothing = "No result"
processResult (Just n) = "Result: " ++ show n
```

## Product Types

### Tuples and Records
```haskell
-- Product type as tuple
type Point = (Double, Double)

-- Product type as record
data Person = Person
    { name :: String
    , age :: Int
    , email :: String
    } deriving (Show, Eq)
```

### Record Syntax Benefits
```haskell
-- Automatic accessor functions
getName :: Person -> String
getName = name

-- Record updates
birthday :: Person -> Person
birthday person = person { age = age person + 1 }
```

## Recursive Data Types

### Lists
```haskell
data List a = Nil | Cons a (List a)

-- Operations on custom lists
listMap :: (a -> b) -> List a -> List b
listMap _ Nil = Nil
listMap f (Cons x xs) = Cons (f x) (listMap f xs)
```

### Trees
```haskell
data Tree a = Empty | Node a (Tree a) (Tree a)

-- Tree operations
treeMap :: (a -> b) -> Tree a -> Tree b
treeMap _ Empty = Empty
treeMap f (Node x left right) = 
    Node (f x) (treeMap f left) (treeMap f right)

treeHeight :: Tree a -> Int
treeHeight Empty = 0
treeHeight (Node _ left right) = 
    1 + max (treeHeight left) (treeHeight right)
```

## Advanced ADT Patterns

### Phantom Types
```haskell
data Temperature scale = Temperature Double

data Celsius
data Fahrenheit

celsiusToFahrenheit :: Temperature Celsius -> Temperature Fahrenheit
celsiusToFahrenheit (Temperature c) = Temperature (c * 9/5 + 32)
```

### Newtype Wrappers
```haskell
newtype UserId = UserId Int
newtype ProductId = ProductId Int

-- Type safety prevents mixing up IDs
getUser :: UserId -> User
getProduct :: ProductId -> Product
```

## Research Papers

### Foundational Work
1. **"Algebraic Data Types in Haskell" (1991)** - Philip Wadler
2. **"Views: A Way for Pattern Matching to Cohabit with Data Abstraction" (1987)** - Philip Wadler

### Advanced Topics
1. **"Generalised Algebraic Data Types and Object-Oriented Programming" (2005)** - Andrew Kennedy and Claudio Russo
2. **"Fun with Type Functions" (2008)** - Oleg Kiselyov et al.

ADTs provide the foundation for creating robust, maintainable Haskell programs by making illegal states unrepresentable.