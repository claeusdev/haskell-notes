# Type Classes and Polymorphism - Comprehensive Notes

## Overview

Type classes are Haskell's elegant solution to ad-hoc polymorphism, allowing you to define generic functions that work across different types while maintaining type safety. They provide a powerful way to organize code and express interfaces without the overhead of object-oriented inheritance, enabling clean, composable, and maintainable code.

**Key Learning Objectives:**
- Master the fundamental type classes and their laws
- Understand parametric vs ad-hoc polymorphism
- Learn to create custom type classes and instances
- Explore advanced type class features and patterns
- Understand type class constraints and their implications
- Apply type classes to solve real-world problems elegantly

## Understanding Polymorphism: The Foundation

### Parametric Polymorphism: One Size Fits All

Parametric polymorphism allows functions to work with any type, providing maximum flexibility while maintaining type safety.

**Basic Parametric Polymorphism:**
```haskell
-- Identity function works with any type
identity :: a -> a
identity x = x

-- Const function with two type parameters
const :: a -> b -> a
const x _ = x

-- Polymorphic data structures
data Pair a b = Pair a b
    deriving (Show, Eq)

-- Polymorphic functions on data structures
swap :: Pair a b -> Pair b a
swap (Pair x y) = Pair y x

-- Higher-order polymorphic functions
map :: (a -> b) -> [a] -> [b]
map _ [] = []
map f (x:xs) = f x : map f xs

-- Constrained parametric polymorphism
elem :: Eq a => a -> [a] -> Bool
elem _ [] = False
elem x (y:ys) = x == y || elem x ys
```

**Benefits of Parametric Polymorphism:**
- **Code Reuse**: Write once, use with many types
- **Type Safety**: Compiler ensures type consistency
- **Performance**: No runtime overhead
- **Composability**: Functions compose naturally

### Ad-Hoc Polymorphism: Type-Specific Behavior

Ad-hoc polymorphism allows different implementations for different types, enabling type-specific behavior while maintaining a common interface.

**Type Classes as Ad-Hoc Polymorphism:**
```haskell
-- Different behavior for different types
class Drawable a where
    draw :: a -> String

-- Different implementations
instance Drawable Circle where
    draw (Circle r) = "Drawing circle with radius " ++ show r

instance Drawable Rectangle where
    draw (Rectangle w h) = "Drawing rectangle " ++ show w ++ "x" ++ show h

-- Same interface, different behavior
drawAll :: Drawable a => [a] -> [String]
drawAll = map draw
```

## Basic Type Classes: The Foundation

### The Eq Type Class: Equality Testing

The `Eq` type class provides equality testing capabilities for types.

**Eq Definition and Laws:**
```haskell
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    
    -- Default implementations (circular)
    x /= y = not (x == y)
    x == y = not (x /= y)

-- Laws that instances must satisfy:
-- 1. Reflexivity: x == x = True
-- 2. Symmetry: x == y = y == x
-- 3. Transitivity: if x == y and y == z then x == z
-- 4. Substitutivity: if x == y then f x == f y
```

**Custom Eq Instances:**
```haskell
-- Custom data type with Eq instance
data Person = Person { name :: String, age :: Int }
    deriving (Show)

-- Manual Eq instance
instance Eq Person where
    (Person name1 age1) == (Person name2 age2) = 
        name1 == name2 && age1 == age2

-- Using deriving for automatic instance
data Color = Red | Green | Blue
    deriving (Eq, Show)

-- Eq with constraints
instance Eq a => Eq (Maybe a) where
    Nothing == Nothing = True
    Nothing == Just _ = False
    Just _ == Nothing = False
    Just x == Just y = x == y
```

### The Ord Type Class: Ordering

The `Ord` type class provides ordering capabilities, building on `Eq`.

**Ord Definition and Laws:**
```haskell
class Eq a => Ord a where
    compare :: a -> a -> Ordering
    (<), (<=), (>), (>=) :: a -> a -> Bool
    max, min :: a -> a -> a
    
    -- Default implementations
    x < y = compare x y == LT
    x <= y = compare x y /= GT
    x > y = compare x y == GT
    x >= y = compare x y /= LT
    max x y = if x >= y then x else y
    min x y = if x <= y then x else y

-- Laws:
-- 1. Antisymmetry: if x <= y and y <= x then x == y
-- 2. Transitivity: if x <= y and y <= z then x <= z
-- 3. Totality: x <= y or y <= x
```

### The Show Type Class: String Representation

The `Show` type class provides string representation capabilities.

**Show Definition:**
```haskell
class Show a where
    show :: a -> String
    showsPrec :: Int -> a -> ShowS
    showList :: [a] -> ShowS
    
    -- Default implementations
    show = showsPrec 0
    showsPrec _ x s = show x ++ s
    showList = showList__

-- ShowS is a function type for efficient string building
type ShowS = String -> String
```

## Custom Type Classes

### Creating Your Own Type Classes
```haskell
-- Serializable type class
class Serializable a where
    serialize :: a -> String
    deserialize :: String -> Maybe a

-- JSON-like type class
class ToJSON a where
    toJSON :: a -> String

class FromJSON a where
    fromJSON :: String -> Maybe a
```

## Parametric Polymorphism

### Polymorphic Functions
```haskell
-- Function that works with any type
identity :: a -> a
identity x = x

-- Function with multiple type parameters
const :: a -> b -> a
const x _ = x

-- Polymorphic data structures
data Pair a b = Pair a b
```

## Advanced Type Classes

### Functor
```haskell
class Functor f where
    fmap :: (a -> b) -> f a -> f b

-- Laws:
-- fmap id = id
-- fmap (f . g) = fmap f . fmap g
```

### Applicative
```haskell
class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b
```

### Monad
```haskell
class Applicative m => Monad m where
    (>>=) :: m a -> (a -> m b) -> m b
    return :: a -> m a
```

## Multi-Parameter Type Classes

### Functional Dependencies
```haskell
{-# LANGUAGE FunctionalDependencies #-}

class Collection c e | c -> e where
    empty :: c
    insert :: e -> c -> c
    member :: e -> c -> Bool
```

### Associated Types
```haskell
{-# LANGUAGE TypeFamilies #-}

class Container c where
    type Element c
    empty :: c
    insert :: Element c -> c -> c
```

## Type Class Constraints

### Using Constraints
```haskell
-- Function requiring Ord constraint
sort :: Ord a => [a] -> [a]

-- Multiple constraints
showAndCompare :: (Show a, Ord a) => a -> a -> String
showAndCompare x y = show x ++ " compared to " ++ show y ++ " is " ++ show (compare x y)
```

## Research Papers

### Foundational Papers
1. **"Type Classes in Haskell" (1988)** - Philip Wadler and Stephen Blott
2. **"How to Make Ad-hoc Polymorphism Less Ad Hoc" (1989)** - Philip Wadler and Stephen Blott
3. **"A System of Constructor Classes" (1993)** - Mark P. Jones

### Advanced Topics
1. **"Type Classes: Exploring the Design Space" (1997)** - Simon Peyton Jones, Mark Jones, Erik Meijer
2. **"Associated Types with Class" (2005)** - Manuel M. T. Chakravarty et al.

Type classes provide a clean, principled way to achieve polymorphism while maintaining Haskell's strong type system guarantees.