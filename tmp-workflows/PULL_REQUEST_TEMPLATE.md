# Pull Request Template

Thank you for contributing to ED-Batch! Please use this template to help us review your pull request effectively.

## Summary
<!-- Provide a brief description of what this PR accomplishes -->

## Type of Change
<!-- Please check all that apply -->
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Code refactoring (no functional changes)
- [ ] Test improvements

## Related Issues
<!-- Link to any related issues -->
Fixes #(issue number)
Closes #(issue number)
Related to #(issue number)

---

## Changes Made
<!-- Provide a detailed list of changes -->
- 
- 
- 

## Technical Details
<!-- Describe the technical approach and any important implementation details -->

### Code Changes
<!-- Describe the main code changes -->
- **Files modified:** 
- **New files added:** 
- **Files removed:** 

### Architecture/Design Changes
<!-- If applicable, describe any architectural or design changes -->

---

## Testing
<!-- Describe the testing you have performed -->

### Test Results
- [ ] All existing tests pass (`make test`)
- [ ] New tests added for new functionality
- [ ] Tests cover edge cases and error conditions
- [ ] Gradient checks pass (for differentiable operations)
- [ ] Performance benchmarks run (if applicable)

### Test Commands Used
```bash
# List the commands you used to test your changes
make test
# Add any additional test commands
```

### Test Coverage
<!-- Describe what your tests cover -->
- [ ] Unit tests for new functionality
- [ ] Integration tests
- [ ] Edge case testing
- [ ] Error condition testing
- [ ] Performance regression testing

### Platforms Tested
<!-- Check all platforms you tested on -->
- [ ] Linux (specify distribution and version)
- [ ] macOS (specify version)
- [ ] Windows (specify version)
- [ ] CPU-only build
- [ ] GPU build (specify CUDA version)

---

## Performance Impact
<!-- Describe any performance implications -->

### Benchmarks
<!-- If applicable, provide benchmark results -->
- **Before:** 
- **After:** 
- **Improvement:** 

### Memory Usage
<!-- If applicable, describe memory usage changes -->

---

## Documentation
<!-- Check all that apply -->
- [ ] Code is self-documenting with clear variable/function names
- [ ] Added comments for complex algorithms or logic
- [ ] Updated API documentation (if applicable)
- [ ] Updated user documentation (if applicable)
- [ ] Added examples for new features
- [ ] Updated changelog (if applicable)

---

## Code Quality Checklist
<!-- Please verify all items before submitting -->

### Code Style
- [ ] Code follows ED-Batch style guidelines (see `doc/source/code_style.rst`)
- [ ] Function names use `snake_case`
- [ ] Proper use of `const` for immutable parameters
- [ ] Appropriate error handling with `DYNET_INVALID_ARG`, `DYNET_RUNTIME_ERR`, `DYNET_ASSERT`
- [ ] No unnecessary memory allocations in hot paths
- [ ] Proper memory management (no manual `new`/`delete`)

### Implementation Quality
- [ ] Code is efficient and follows ED-Batch performance guidelines
- [ ] No code duplication
- [ ] Functions are focused and have single responsibility
- [ ] Proper handling of edge cases
- [ ] Thread-safe (if applicable)

### Testing Quality
- [ ] Tests are comprehensive and cover the main functionality
- [ ] Tests include both positive and negative cases
- [ ] Tests are maintainable and well-structured
- [ ] No flaky or unreliable tests

---

## Backward Compatibility
<!-- Describe any backward compatibility considerations -->
- [ ] No breaking changes to existing API
- [ ] Existing models/code will continue to work
- [ ] Migration guide provided (if breaking changes are necessary)

---

## System Information
<!-- Provide information about your development environment -->

**Operating System:** 
<!-- e.g., Ubuntu 20.04 -->

**Compiler:** 
<!-- e.g., GCC 9.4.0 -->

**CMake Version:** 
<!-- e.g., 3.16.3 -->

**CUDA Version (if applicable):** 
<!-- e.g., 11.1 -->

**ED-Batch Base Commit:** 
<!-- The commit hash you based your changes on -->

---

## Additional Context
<!-- Any additional information that reviewers should know -->

### Screenshots/Outputs
<!-- If applicable, add screenshots or example outputs -->

### Dependencies
<!-- List any new dependencies added -->

### Future Work
<!-- Any follow-up work or known limitations -->

---

## Reviewer Notes
<!-- Any specific areas you'd like reviewers to focus on -->

---

## Final Checklist
<!-- Please check all items before submitting -->
- [ ] I have read the [Feature Request and Bug Fix Walkthrough](doc/source/feature_request_walkthrough.rst)
- [ ] My code follows the ED-Batch style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes (`make test`)
- [ ] Any dependent changes have been merged and published

---

**Note:** Please ensure you have followed our [contribution guidelines](doc/source/contributing.rst) and completed all items in this checklist before requesting a review.
