//! Tokenizer that converts a mathematical expression in a string form into a series of `Token`s.
//!
//! The underlying parser is build using the [nom] parser combinator crate.
//!
//! The parser should tokenize only well-formed expressions.
//!
//! [nom]: https://crates.io/crates/nom

use nom::Err;
use nom::IResult;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::multispace1;
use nom::combinator::{map, map_res, opt};
use nom::error::{Error, ErrorKind};
use nom::sequence::{delimited, preceded, terminated};
use std::fmt;
use std::str::from_utf8;

/// An error reported by the parser.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// A token that is not allowed at the given location (contains the location of the offending
    /// character in the source string).
    UnexpectedToken(usize),
    /// Missing right parentheses at the end of the source string (contains the number of missing
    /// parens).
    MissingRParen(i32),
    /// Missing operator or function argument at the end of the expression.
    MissingArgument,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ParseError::UnexpectedToken(i) => write!(f, "Unexpected token at byte {}.", i),
            ParseError::MissingRParen(i) => write!(
                f,
                "Missing {} right parenthes{}.",
                i,
                if i == 1 { "is" } else { "es" }
            ),
            ParseError::MissingArgument => write!(f, "Missing argument at the end of expression."),
        }
    }
}

impl std::error::Error for ParseError {
    fn description(&self) -> &str {
        match *self {
            ParseError::UnexpectedToken(_) => "unexpected token",
            ParseError::MissingRParen(_) => "missing right parenthesis",
            ParseError::MissingArgument => "missing argument",
        }
    }
}

/// Mathematical operations.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Operation {
    Plus,
    Minus,
    Times,
    Div,
    Rem,
    Pow,
    Fact,
}

/// Expression tokens.
#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    /// Binary operation.
    Binary(Operation),
    /// Unary operation.
    Unary(Operation),

    /// Left parenthesis.
    LParen,
    /// Right parenthesis.
    RParen,
    /// Comma: function argument separator
    Comma,

    /// A number.
    Number(f64),
    /// A variable.
    Var(String),
    /// A function with name and number of arguments.
    Func(String, Option<usize>),
}

fn binop(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;
    alt((
        map(tag("+"), |_| Token::Binary(Operation::Plus)),
        map(tag("-"), |_| Token::Binary(Operation::Minus)),
        map(tag("*"), |_| Token::Binary(Operation::Times)),
        map(tag("/"), |_| Token::Binary(Operation::Div)),
        map(tag("%"), |_| Token::Binary(Operation::Rem)),
        map(tag("^"), |_| Token::Binary(Operation::Pow)),
    ))
    .parse(input)
}

fn negpos(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;
    alt((
        map(tag("+"), |_| Token::Unary(Operation::Plus)),
        map(tag("-"), |_| Token::Unary(Operation::Minus)),
    ))
    .parse(input)
}

fn fact(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;
    map(tag("!"), |_| Token::Unary(Operation::Fact)).parse(input)
}

fn lparen(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;
    map(tag("("), |_| Token::LParen).parse(input)
}

fn rparen(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;
    map(tag(")"), |_| Token::RParen).parse(input)
}

fn comma(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;
    map(tag(","), |_| Token::Comma).parse(input)
}

/// Parse an identifier:
///
/// Must start with a letter or an underscore, can be followed by letters, digits or underscores.
fn ident(input: &[u8]) -> IResult<&[u8], &[u8]> {
    // first character must be 'a'..='z' | 'A'..='Z' | '_'
    match input.first().cloned() {
        Some(b'a'..=b'z') | Some(b'A'..=b'Z') | Some(b'_') => {
            let n = input
                .iter()
                .skip(1)
                .take_while(|&&c| matches!(c, b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'0'..=b'9'))
                .count();
            let (parsed, rest) = input.split_at(n + 1);
            Ok((rest, parsed))
        }
        None => Err(Err::Incomplete(nom::Needed::new(1))),
        _ => Err(Err::Error(Error::new(input, ErrorKind::Satisfy))),
    }
}

fn var(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;
    map(map_res(ident, from_utf8), |s: &str| Token::Var(s.into())).parse(input)
}

// Parse `func(`, returns `func`.
fn func(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;
    map(
        map_res(
            terminated(ident, preceded(opt(multispace1), tag("("))),
            from_utf8,
        ),
        |s: &str| Token::Func(s.into(), None),
    )
    .parse(input)
}

/// Matches one or more digit characters `0`...`9`.
///
/// Never returns `nom::IResult::Incomplete`.
///
/// Fix of IMHO broken `nom::digit`, which parses an empty string successfully.
fn digit_complete(input: &[u8]) -> IResult<&[u8], &[u8]> {
    let n = input.iter().take_while(|&&c| c.is_ascii_digit()).count();
    if n > 0 {
        let (parsed, rest) = input.split_at(n);
        Ok((rest, parsed))
    } else {
        Err(Err::Error(Error::new(input, ErrorKind::Digit)))
    }
}

/// Parser that matches a floating point number and returns the total length consumed.
fn float(input: &[u8]) -> IResult<&[u8], usize> {
    use nom::Parser;

    let (input, integer_part) = digit_complete(input)?;
    let (input, decimal_part) = opt((tag("."), opt(digit_complete))).parse(input)?;
    let (input, exp_part) = exp(input)?;

    let total_len = integer_part.len()
        + decimal_part
            .map(|(_dot, digits)| 1 + digits.map(|d| d.len()).unwrap_or(0))
            .unwrap_or(0)
        + exp_part.unwrap_or(0);

    Ok((input, total_len))
}

/// Parser that matches the exponential part of a float. If the `input[0] == 'e' | 'E'` then at
/// least one digit must match.
fn exp(input: &[u8]) -> IResult<&[u8], Option<usize>> {
    use nom::Parser;

    let e_result: IResult<&[u8], &[u8]> = alt((tag("e"), tag("E"))).parse(input);
    match e_result {
        Err(_) => Ok((input, None)),
        Ok((rest, _)) => {
            let (remaining, sign) = opt(alt((tag("+"), tag("-")))).parse(rest)?;
            let (remaining, digits) = digit_complete(remaining)?;
            let len = 1 + sign.map(|s| s.len()).unwrap_or(0) + digits.len();
            Ok((remaining, Some(len)))
        }
    }
}

fn number(input: &[u8]) -> IResult<&[u8], Token> {
    use std::str::FromStr;

    match float(input) {
        Ok((rest, l)) => {
            // it should be safe to call unwrap here instead of the error checking, since
            // `float` should match only well-formed numbers
            from_utf8(&input[..l])
                .ok()
                .and_then(|s| f64::from_str(s).ok())
                .map_or(Err(Err::Error(Error::new(input, ErrorKind::Float))), |f| {
                    Ok((rest, Token::Number(f)))
                })
        }
        Err(e) => Err(e),
    }
}

fn lexpr(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;

    delimited(
        opt(multispace1),
        alt((number, func, var, negpos, lparen)),
        opt(multispace1),
    )
    .parse(input)
}

fn after_rexpr(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;

    delimited(
        opt(multispace1),
        alt((fact, binop, rparen)),
        opt(multispace1),
    )
    .parse(input)
}

fn after_rexpr_no_paren(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;

    delimited(opt(multispace1), alt((fact, binop)), opt(multispace1)).parse(input)
}

fn after_rexpr_comma(input: &[u8]) -> IResult<&[u8], Token> {
    use nom::Parser;

    delimited(
        opt(multispace1),
        alt((fact, binop, rparen, comma)),
        opt(multispace1),
    )
    .parse(input)
}

#[derive(Debug, Clone, Copy)]
enum TokenizerState {
    // accept any token that is an expression from the left: var, num, (, negpos
    LExpr,
    // accept any token that needs an expression on the left: fact, binop, ), comma
    AfterRExpr,
}

#[derive(Debug, Clone, Copy)]
enum ParenState {
    Subexpr,
    Func,
}

/// Tokenize a given mathematical expression.
///
/// The parser should return `Ok` only if the expression is well-formed.
///
/// # Failure
///
/// Returns `Err` if the expression is not well-formed.
pub fn tokenize<S: AsRef<str>>(input: S) -> Result<Vec<Token>, ParseError> {
    use self::TokenizerState::*;
    use nom::Err;

    let mut state = LExpr;
    // number of function arguments left
    let mut paren_stack = vec![];

    let mut res = vec![];

    let input = input.as_ref().as_bytes();
    let mut s = input;

    while !s.is_empty() {
        let r = match (state, paren_stack.last()) {
            (LExpr, _) => lexpr(s),
            (AfterRExpr, None) => after_rexpr_no_paren(s),
            (AfterRExpr, Some(&ParenState::Subexpr)) => after_rexpr(s),
            (AfterRExpr, Some(&ParenState::Func)) => after_rexpr_comma(s),
        };

        match r {
            Ok((rest, t)) => {
                match t {
                    Token::LParen => {
                        paren_stack.push(ParenState::Subexpr);
                    }
                    Token::Func(..) => {
                        paren_stack.push(ParenState::Func);
                    }
                    Token::RParen => {
                        paren_stack.pop().expect("The paren_stack is empty!");
                    }
                    Token::Var(_) | Token::Number(_) => {
                        state = AfterRExpr;
                    }
                    Token::Binary(_) | Token::Comma => {
                        state = LExpr;
                    }
                    _ => {}
                }
                res.push(t);
                s = rest;
            }
            Err(Err::Error(Error { input: p, .. })) => {
                let offset = input.len() - p.len();
                return Err(ParseError::UnexpectedToken(offset));
            }
            Err(_) => {
                panic!(
                    "Unexpected parse result when parsing `{}` at `{}`: {:?}",
                    String::from_utf8_lossy(input),
                    String::from_utf8_lossy(s),
                    r
                );
            }
        }
    }

    match state {
        LExpr => Err(ParseError::MissingArgument),
        _ if !paren_stack.is_empty() => Err(ParseError::MissingRParen(paren_stack.len() as i32)),
        _ => Ok(res),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{binop, func, number, var};

    #[test]
    fn it_works() {
        assert_eq!(binop(b"+"), Ok((&b""[..], Token::Binary(Operation::Plus))));
        assert_eq!(number(b"32143"), Ok((&b""[..], Token::Number(32143f64))));
        assert_eq!(var(b"abc"), Ok((&b""[..], Token::Var("abc".into()))));
        assert_eq!(
            func(b"abc("),
            Ok((&b""[..], Token::Func("abc".into(), None)))
        );
        assert_eq!(
            func(b"abc ("),
            Ok((&b""[..], Token::Func("abc".into(), None)))
        );
    }

    #[test]
    fn test_var() {
        for &s in ["abc", "U0", "_034", "a_be45EA", "aAzZ_"].iter() {
            assert_eq!(var(s.as_bytes()), Ok((&b""[..], Token::Var(s.into()))));
        }

        // Just check that these return error - exact error type has changed in nom 8.0
        assert!(var(b"").is_err());
        assert!(var(b"0").is_err());
    }

    #[test]
    fn test_func() {
        for &s in ["abc(", "u0(", "_034 (", "A_be45EA  ("].iter() {
            assert_eq!(
                func(s.as_bytes()),
                Ok((
                    &b""[..],
                    Token::Func((&s[0..s.len() - 1]).trim().into(), None)
                ))
            );
        }

        // Just check that these return error - exact error type has changed in nom 8.0
        assert!(func(b"").is_err());
        assert!(func(b"(").is_err());
        assert!(func(b"0(").is_err());
    }

    #[test]
    fn test_number() {
        assert_eq!(number(b"32143"), Ok((&b""[..], Token::Number(32143f64))));
        assert_eq!(number(b"2."), Ok((&b""[..], Token::Number(2.0f64))));
        assert_eq!(
            number(b"32143.25"),
            Ok((&b""[..], Token::Number(32143.25f64)))
        );
        assert_eq!(
            number(b"0.125e9"),
            Ok((&b""[..], Token::Number(0.125e9f64)))
        );
        assert_eq!(
            number(b"20.5E-3"),
            Ok((&b""[..], Token::Number(20.5E-3f64)))
        );
        assert_eq!(
            number(b"123423e+50"),
            Ok((&b""[..], Token::Number(123423e+50f64)))
        );

        // Just check that these return error - exact error type has changed in nom 8.0
        assert!(number(b"").is_err());
        assert!(number(b".2").is_err());
        assert!(number(b"+").is_err());
        assert!(number(b"e").is_err());
        assert!(number(b"1E").is_err());
        assert!(number(b"1e+").is_err());
    }

    #[test]
    fn test_tokenize() {
        use super::Operation::*;
        use super::Token::*;

        assert_eq!(tokenize("a"), Ok(vec![Var("a".into())]));

        assert_eq!(
            tokenize("2 +(3--2) "),
            Ok(vec![
                Number(2f64),
                Binary(Plus),
                LParen,
                Number(3f64),
                Binary(Minus),
                Unary(Minus),
                Number(2f64),
                RParen
            ])
        );

        assert_eq!(
            tokenize("-2^ ab0 *12 - C_0"),
            Ok(vec![
                Unary(Minus),
                Number(2f64),
                Binary(Pow),
                Var("ab0".into()),
                Binary(Times),
                Number(12f64),
                Binary(Minus),
                Var("C_0".into()),
            ])
        );

        assert_eq!(
            tokenize("-sin(pi * 3)^ cos(2) / Func2(x, f(y), z) * _buildIN(y)"),
            Ok(vec![
                Unary(Minus),
                Func("sin".into(), None),
                Var("pi".into()),
                Binary(Times),
                Number(3f64),
                RParen,
                Binary(Pow),
                Func("cos".into(), None),
                Number(2f64),
                RParen,
                Binary(Div),
                Func("Func2".into(), None),
                Var("x".into()),
                Comma,
                Func("f".into(), None),
                Var("y".into()),
                RParen,
                Comma,
                Var("z".into()),
                RParen,
                Binary(Times),
                Func("_buildIN".into(), None),
                Var("y".into()),
                RParen,
            ])
        );

        assert_eq!(
            tokenize("2 % 3"),
            Ok(vec![Number(2f64), Binary(Rem), Number(3f64)])
        );

        assert_eq!(
            tokenize("1 + 3! + 1"),
            Ok(vec![
                Number(1f64),
                Binary(Plus),
                Number(3f64),
                Unary(Fact),
                Binary(Plus),
                Number(1f64)
            ])
        );

        assert_eq!(tokenize("!3"), Err(ParseError::UnexpectedToken(0)));

        assert_eq!(tokenize("()"), Err(ParseError::UnexpectedToken(1)));

        assert_eq!(tokenize(""), Err(ParseError::MissingArgument));
        assert_eq!(tokenize("2)"), Err(ParseError::UnexpectedToken(1)));
        assert_eq!(tokenize("2^"), Err(ParseError::MissingArgument));
        assert_eq!(tokenize("(((2)"), Err(ParseError::MissingRParen(2)));
        assert_eq!(tokenize("f(2,)"), Err(ParseError::UnexpectedToken(4)));
        assert_eq!(tokenize("f(,2)"), Err(ParseError::UnexpectedToken(2)));
    }
}
