# Algebraic Data Types - Comprehensive Notes

## Overview

Algebraic Data Types (ADTs) are one of Haskell's most powerful features, allowing you to create precise, type-safe representations of your problem domain. This comprehensive guide covers sum types, product types, recursive types, and advanced ADT patterns that form the foundation of robust, maintainable Haskell programs.

**Key Learning Objectives:**
- Master sum types (union types) and their applications
- Understand product types and record syntax
- Learn recursive data types and their patterns
- Explore advanced ADT techniques like phantom types and GADTs
- Understand performance characteristics of different ADT patterns
- Apply ADTs to solve real-world problems with type safety

## Sum Types (Union Types): Representing Alternatives

Sum types represent data that can be one of several alternatives. They're called "sum" types because the total number of possible values is the sum of the values from each alternative.

### Basic Sum Types: The Foundation

**Simple Sum Types:**
```haskell
-- Basic enumeration
data Color = Red | Green | Blue
    deriving (Show, Eq, Ord)

-- Boolean as a sum type
data Bool = True | False
    deriving (Show, Eq, Ord)

-- Maybe type for optional values
data Maybe a = Nothing | Just a
    deriving (Show, Eq, Ord)

-- Either type for success/failure
data Either a b = Left a | Right b
    deriving (Show, Eq, Ord)

-- Ordering type for comparisons
data Ordering = LT | EQ | GT
    deriving (Show, Eq, Ord)
```

**Benefits of Sum Types:**
- **Type Safety**: Compiler ensures all cases are handled
- **Exhaustiveness**: Pattern matching must cover all alternatives
- **Expressiveness**: Model domain concepts precisely
- **Maintainability**: Adding new cases requires updating all pattern matches

### Pattern Matching on Sum Types: Exhaustive Coverage

Pattern matching on sum types is the primary way to work with them, and Haskell ensures all cases are covered.

**Basic Pattern Matching:**
```haskell
-- Simple pattern matching
colorName :: Color -> String
colorName Red = "red"
colorName Green = "green"
colorName Blue = "blue"

-- Pattern matching with guards
colorRGB :: Color -> (Int, Int, Int)
colorRGB Red = (255, 0, 0)
colorRGB Green = (0, 255, 0)
colorRGB Blue = (0, 0, 255)

-- Pattern matching on Maybe
processResult :: Maybe Int -> String
processResult Nothing = "No result"
processResult (Just n) = "Result: " ++ show n

-- Pattern matching on Either
handleResult :: Either String Int -> String
handleResult (Left error) = "Error: " ++ error
handleResult (Right value) = "Success: " ++ show value
```

**Advanced Pattern Matching:**
```haskell
-- Nested pattern matching
processNested :: Maybe (Either String Int) -> String
processNested Nothing = "No value"
processNested (Just (Left error)) = "Error: " ++ error
processNested (Just (Right value)) = "Value: " ++ show value

-- Pattern matching with as-patterns
processWithAs :: Maybe Int -> String
processWithAs Nothing = "No value"
processWithAs result@(Just n) = 
    "Got result " ++ show result ++ " with value " ++ show n

-- Pattern matching with guards
classifyNumber :: Maybe Int -> String
classifyNumber Nothing = "No number"
classifyNumber (Just n)
    | n < 0 = "Negative: " ++ show n
    | n == 0 = "Zero"
    | n > 0 = "Positive: " ++ show n
```

## Product Types: Combining Data

Product types represent data that contains multiple values simultaneously. They're called "product" types because the total number of possible values is the product of the values from each component.

### Tuples: Simple Product Types

**Basic Tuples:**
```haskell
-- Pair (2-tuple)
type Point2D = (Double, Double)
type NameAge = (String, Int)

-- Triple (3-tuple)
type Point3D = (Double, Double, Double)
type PersonInfo = (String, Int, String)  -- name, age, email

-- Quadruple (4-tuple)
type Rectangle = (Double, Double, Double, Double)  -- x, y, width, height
```

**Working with Tuples:**
```haskell
-- Tuple construction
point :: Point2D
point = (3.0, 4.0)

-- Tuple access
getX :: Point2D -> Double
getX (x, _) = x

getY :: Point2D -> Double
getY (_, y) = y

-- Tuple operations
distance :: Point2D -> Point2D -> Double
distance (x1, y1) (x2, y2) = sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Tuple transformation
swap :: (a, b) -> (b, a)
swap (x, y) = (y, x)

-- Built-in tuple functions
first :: (a, b) -> a
first = fst

second :: (a, b) -> b
second = snd
```

### Records: Named Product Types

Records provide a more structured way to define product types with named fields.

**Basic Record Syntax:**
```haskell
-- Simple record
data Person = Person
    { name :: String
    , age :: Int
    , email :: String
    } deriving (Show, Eq, Ord)

-- Record construction
john :: Person
john = Person "John Doe" 30 "john@example.com"

-- Field access
getName :: Person -> String
getName = name

getAge :: Person -> Int
getAge = age

-- Record updates
birthday :: Person -> Person
birthday person = person { age = age person + 1 }

-- Multiple field updates
updateContact :: Person -> String -> Person
updateContact person newEmail = person { email = newEmail }

-- Record pattern matching
isAdult :: Person -> Bool
isAdult (Person _ age _) = age >= 18

-- Named field pattern matching
isAdultNamed :: Person -> Bool
isAdultNamed Person { age = a } = a >= 18
```

**Advanced Record Patterns:**
```haskell
-- Records with default values
data Config = Config
    { host :: String
    , port :: Int
    , timeout :: Int
    , debug :: Bool
    } deriving (Show, Eq)

-- Default configuration
defaultConfig :: Config
defaultConfig = Config
    { host = "localhost"
    , port = 8080
    , timeout = 30
    , debug = False
    }

-- Partial updates
updateHost :: String -> Config -> Config
updateHost newHost config = config { host = newHost }

-- Record with optional fields
data User = User
    { userId :: Int
    , username :: String
    , email :: Maybe String
    , profile :: Maybe String
    } deriving (Show, Eq)

-- Safe field access
getEmail :: User -> String
getEmail User { email = Just e } = e
getEmail User { email = Nothing } = "No email"
```

### Record Syntax Benefits and Best Practices

**Automatic Accessor Functions:**
```haskell
-- Records automatically generate accessor functions
data Point = Point { x :: Double, y :: Double } deriving (Show)

-- These functions are automatically generated:
-- x :: Point -> Double
-- y :: Point -> Double

-- Usage
origin :: Point
origin = Point 0.0 0.0

getX :: Point -> Double
getX = x  -- Using the generated accessor

-- Record updates
moveRight :: Double -> Point -> Point
moveRight distance point = point { x = x point + distance }
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