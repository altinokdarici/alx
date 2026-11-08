//! What a turn costs at the providers' published API prices.

use alx_protocol::Usage;

/// USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Price {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

impl Price {
    /// Anthropic's scheme: cache reads at 0.1x and 5-minute cache writes at
    /// 1.25x the input price.
    const fn anthropic(input: f64, output: f64) -> Self {
        Self {
            input,
            output,
            cache_read: input * 0.1,
            cache_write: input * 1.25,
        }
    }

    pub fn cost(&self, u: &Usage) -> f64 {
        (u.input_tokens as f64 * self.input
            + u.output_tokens as f64 * self.output
            + u.cache_read_tokens as f64 * self.cache_read
            + u.cache_write_tokens as f64 * self.cache_write)
            / 1_000_000.0
    }
}

/// Model-name prefixes and their prices; the first match wins, so more
/// specific prefixes come first. Prefixes end in `-` so they match
/// Anthropic API names only, not Copilot's dotted ones (`claude-haiku-4.5`).
const PRICES: &[(&str, Price)] = &[
    ("claude-opus-4-1", Price::anthropic(15.0, 75.0)),
    ("claude-opus-4-0", Price::anthropic(15.0, 75.0)),
    ("claude-opus-4-2025", Price::anthropic(15.0, 75.0)),
    ("claude-opus-4-", Price::anthropic(5.0, 25.0)),
    ("claude-sonnet-4-", Price::anthropic(3.0, 15.0)),
    ("claude-3-7-sonnet-", Price::anthropic(3.0, 15.0)),
    ("claude-3-5-sonnet-", Price::anthropic(3.0, 15.0)),
    ("claude-haiku-4-", Price::anthropic(1.0, 5.0)),
    ("claude-3-5-haiku-", Price::anthropic(0.8, 4.0)),
];

/// The price of `model`, if known. Unknown models have no cost shown rather
/// than a guessed one.
pub fn price(model: &str) -> Option<Price> {
    PRICES
        .iter()
        .find(|(prefix, _)| model.starts_with(prefix))
        .map(|(_, p)| *p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prices_by_most_specific_prefix() {
        assert_eq!(price("claude-opus-4-1-20250805").unwrap().input, 15.0);
        assert_eq!(price("claude-opus-4-5").unwrap().input, 5.0);
        assert_eq!(price("claude-opus-4-20250514").unwrap().input, 15.0);
        assert!(price("gpt-5").is_none());
        // Copilot's model names (dotted) are subscription-billed, not priced.
        assert!(price("claude-haiku-4.5").is_none());
    }

    #[test]
    fn cache_tokens_are_priced_separately() {
        let p = price("claude-haiku-4-5-20251001").unwrap();
        let u = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 1_000_000,
        };
        assert!((p.cost(&u) - (1.0 + 5.0 + 0.1 + 1.25)).abs() < 1e-9);
    }
}
