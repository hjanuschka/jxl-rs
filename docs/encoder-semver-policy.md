# Encoder API semver policy

This project follows semantic versioning for the public high-level encoder API.

## Policy

- Additive API changes (new enum variants, new methods, new fields on `#[non_exhaustive]` types) are allowed in minor releases.
- Breaking API changes (signature/type changes or removals) require a major release.
- Deprecated APIs remain available for at least one minor release before removal in the next major release.

## Covered encoder symbols

- `encode::JxlEncoder`
- `encode::JxlEncoderOptions`
- `encode::JxlEncoderMode`
- `encode::JxlEncoderImageData`
- High-level encode methods on `JxlEncoder`
