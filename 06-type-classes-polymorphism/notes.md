# Type Classes and Polymorphism - Notes

## Overview

Type classes are Haskell's solution to ad-hoc polymorphism, allowing you to define generic functions that work across different types while maintaining type safety. They provide a powerful way to organize code and express interfaces without the overhead of object-oriented inheritance.

## Basic Type Classes

### The Eq Type Class
```haskell
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    -- Default implementations
    x /= y = not (x == y)
    x == y = not (x /= y)
```

### The Ord Type Class
```haskell
class Eq a => Ord a where
    compare :: a -> a -> Ordering
    (<), (<=), (>), (>=) :: a -> a -> Bool
    max, min :: a -> a -> a
```

### The Show Type Class
```haskell
class Show a where
    show :: a -> String
    showsPrec :: Int -> a -> ShowS
    showList :: [a] -> ShowS
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