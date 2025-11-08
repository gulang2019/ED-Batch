.. _feature_request_walkthrough:

Feature Request and Bug Fix Walkthrough
========================================

This document provides a comprehensive step-by-step guide for handling feature requests and bug fixes in ED-Batch. It covers the entire process from identifying issues to merging pull requests, serving both contributors and maintainers.

Overview
--------

ED-Batch is a DyNet extension that implements efficient automatic batching of dynamic deep neural networks via finite state machines. The codebase is organized into two main parts:

* **dynet/**: Runtime extension implementation with state machine guided dynamic batching
* **src/**: Static subgraph optimization tools and utilities
* **tests/**: Unit tests and testing infrastructure
* **examples/**: Model implementations and usage examples

1. Identifying and Reporting Issues
-----------------------------------

Types of Issues
~~~~~~~~~~~~~~~

**Bug Reports**
  Issues where existing functionality doesn't work as expected, including:
  
  * Runtime errors or crashes
  * Incorrect computation results
  * Memory leaks or performance regressions
  * Build or installation failures

**Feature Requests**
  Proposals for new functionality, including:
  
  * New operations or node types
  * Performance optimizations
  * API improvements
  * Documentation enhancements

How to Report Issues
~~~~~~~~~~~~~~~~~~~~

Before reporting an issue:

1. **Search existing issues** to avoid duplicates
2. **Test with the latest version** to ensure the issue still exists
3. **Create a minimal reproducible example** when possible

When creating an issue, include:

* **Clear description** of the problem or requested feature
* **Steps to reproduce** (for bugs)
* **Expected vs. actual behavior**
* **System information**: OS, compiler version, CUDA version (if applicable)
* **Code example** demonstrating the issue
* **Error messages** or stack traces (if applicable)

Example Bug Report Template::

    **Bug Description**
    Brief description of what's wrong.

    **Steps to Reproduce**
    1. Step one
    2. Step two
    3. Step three

    **Expected Behavior**
    What should happen.

    **Actual Behavior**
    What actually happens.

    **System Information**
    - OS: Ubuntu 20.04
    - Compiler: GCC 9.4.0
    - CUDA: 11.1 (if applicable)
    - ED-Batch version: commit hash

    **Code Example**
    ```cpp
    // Minimal code that reproduces the issue
    ```

2. Setting Up Development Environment
-------------------------------------

Prerequisites
~~~~~~~~~~~~~

Before contributing to ED-Batch, ensure you have:

* **Eigen 3.4.0**: Required linear algebra library
* **CMake >= 3.12**: Build system
* **C++ compiler**: GCC 7+ or Clang 6+
* **Git**: Version control
* **Optional**: CUDA 11.1+ and cuDNN 8+ for GPU support

Environment Setup
~~~~~~~~~~~~~~~~~

1. **Install Eigen 3.4.0**::

    wget https://fossies.org/linux/privat/eigen-3.4.0.tar.bz2
    tar xjvf eigen-3.4.0.tar.bz2
    export EIGEN_BASE_DIR=${PWD}/eigen-3.4.0

2. **Clone the repository**::

    git clone https://github.com/gulang2019/ED-Batch.git
    cd ED-Batch

3. **Create a development branch**::

    git checkout -b feature/your-feature-name
    # or
    git checkout -b bugfix/issue-description

4. **Build ED-Batch**::

    mkdir build && cd build
    
    # CPU-only build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. \
             -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    
    # GPU build (if CUDA available)
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. \
             -DBACKEND=cuda -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    
    make -j4 && make install
    cd ..
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/lib

5. **Verify installation**::

    make test

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Keep your fork updated**::

    git remote add upstream https://github.com/gulang2019/ED-Batch.git
    git fetch upstream
    git checkout main
    git merge upstream/main

2. **Create feature branches** for each contribution
3. **Make small, focused commits** with clear messages
4. **Test frequently** during development

3. Making Code Changes
----------------------

Code Organization
~~~~~~~~~~~~~~~~~

Understanding the codebase structure is crucial for effective contributions:

**dynet/ Directory**
  Core runtime implementation:
  
  * ``ooc-block.h/cc``: Static subgraph optimization
  * ``ooc-executor.h/cc``: Runtime driver and execution engine
  * ``ooc-scheduler.h/cc``: Dynamic batching algorithms
  * ``nodes-*.h/cc``: Operation implementations
  * ``graph.h/cc``: Computation graph management

**src/ Directory**
  Static optimization tools:
  
  * ``OoC.h/cc``: Pattern cache and optimization utilities
  * ``pq-trees/``: PQ-tree data structures for optimization
  * ``scheduler.cc``: Static scheduling algorithms

**tests/ Directory**
  Unit tests and testing infrastructure:
  
  * ``test-*.cc``: Individual test suites
  * ``test.h``: Common testing utilities

**examples/ Directory**
  Model implementations and usage examples

Common Contribution Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Adding a New Operation**

1. **Define the operation** in appropriate ``nodes-*.h`` file::

    class YourNewOperation : public Node {
    public:
      explicit YourNewOperation(const std::vector<VariableIndex>& a) : Node(a) {}
      virtual bool supports_multibatch() const override { return true; }
      virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
      virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
      virtual void forward_dev_impl(const MyDevice& dev, const std::vector<const Tensor*>& xs, Tensor& fx) const override;
      virtual void backward_dev_impl(const MyDevice& dev,
                                   const std::vector<const Tensor*>& xs,
                                   const Tensor& fx,
                                   const Tensor& dEdf,
                                   unsigned i,
                                   Tensor& dEdxi) const override;
      virtual Dim dim_forward(const std::vector<Dim>& xs) const override;
    };

2. **Implement the operation** in corresponding ``.cc`` file::

    void YourNewOperation::forward_dev_impl(const MyDevice& dev, 
                                           const std::vector<const Tensor*>& xs, 
                                           Tensor& fx) const {
      // Implementation here
      // Remember: fx is not initialized, must be set completely
    }

    void YourNewOperation::backward_dev_impl(const MyDevice& dev,
                                           const std::vector<const Tensor*>& xs,
                                           const Tensor& fx,
                                           const Tensor& dEdf,
                                           unsigned i,
                                           Tensor& dEdxi) const {
      // Implementation here
      // Remember: dEdxi must ACCUMULATE (use += not =)
    }

3. **Add expression interface** in ``expr.h``::

    Expression your_new_operation(const Expression& x);

4. **Implement expression interface** in ``expr.cc``::

    Expression your_new_operation(const Expression& x) {
      return Expression(x.pg, x.pg->add_function<YourNewOperation>({x.i}));
    }

**Fixing a Bug in Existing Code**

1. **Identify the root cause** through debugging and testing
2. **Write a test** that reproduces the bug::

    // In appropriate test-*.cc file
    void test_bug_reproduction() {
      ComputationGraph cg;
      // Set up scenario that triggers the bug
      // Assert expected vs actual behavior
      BOOST_CHECK_EQUAL(expected_result, actual_result);
    }

3. **Implement the fix** following existing code patterns
4. **Verify the test passes** after the fix
5. **Run full test suite** to ensure no regressions

**Performance Optimization**

1. **Profile the code** to identify bottlenecks
2. **Benchmark before and after** changes
3. **Consider memory allocation patterns** (ED-Batch manages its own memory)
4. **Test with different batch sizes** and model configurations

Coding Standards
~~~~~~~~~~~~~~~~

Follow the established coding standards documented in :ref:`code_style`. Key points:

* **Function names**: Use ``snake_case``
* **Memory management**: Avoid allocations in hot paths
* **Error handling**: Use ``DYNET_INVALID_ARG``, ``DYNET_RUNTIME_ERR``, ``DYNET_ASSERT``
* **const correctness**: Always use const for immutable parameters
* **Documentation**: Add clear comments for complex algorithms

Example of well-structured code::

    void efficient_operation(const Tensor& input, Tensor& output) const {
      DYNET_ASSERT(input.d == output.d, "Dimension mismatch in efficient_operation");
      
      const float* input_data = input.v;
      float* output_data = output.v;
      
      // Efficient implementation avoiding temporary allocations
      for (unsigned i = 0; i < input.d.size(); ++i) {
        output_data[i] = some_function(input_data[i]);
      }
    }

4. Testing Procedures
---------------------

ED-Batch uses a comprehensive testing framework to ensure code quality and prevent regressions.

Running Tests
~~~~~~~~~~~~~

**Full Test Suite**::

    make test

This command runs all unit tests and is the primary way to verify your changes don't break existing functionality.

**Individual Test Suites**::

    # Run specific test categories
    ./build/tests/test-nodes     # Test node operations
    ./build/tests/test-exec      # Test execution engine
    ./build/tests/test-rnn       # Test RNN functionality

**Benchmark Tests**::

    cd benchmark
    make
    
    # CPU tests
    ./test_graph tree_lstm CPU 32 128 0 result --dynet-devices CPU
    ./test_block lstm CPU 32 128 0 result --dynet-devices CPU --dynet-autobatch 1
    
    # GPU tests (if available)
    ./test_graph tree_lstm GPU 32 128 0 result --dynet-devices GPU:0
    ./test_block lstm GPU 32 128 0 result --dynet-devices GPU:0 --dynet-autobatch 1

Writing Tests
~~~~~~~~~~~~~

**Unit Test Structure**

Tests are organized in the ``tests/`` directory. Use ``test-dynet.cc`` as a reference for test structure::

    #include "test.h"
    
    void test_your_new_feature() {
      ComputationGraph cg;
      
      // Set up test scenario
      Expression x = input(cg, {2, 3}, {1, 2, 3, 4, 5, 6});
      Expression y = your_new_operation(x);
      
      // Compute and verify results
      cg.forward(y);
      vector<float> expected = {expected_values};
      vector<float> actual = as_vector(y.value());
      
      BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                   actual.begin(), actual.end());
    }

**Test Categories**

* **Correctness tests**: Verify operations produce expected results
* **Edge case tests**: Test boundary conditions and error handling
* **Performance tests**: Ensure optimizations don't regress performance
* **Memory tests**: Check for leaks and proper cleanup

**Best Practices for Testing**

1. **Test both forward and backward passes** for new operations
2. **Include gradient checks** for differentiable operations::

    void test_gradient_check() {
      ComputationGraph cg;
      Expression x = parameter(cg, p);  // some parameter
      Expression y = your_operation(x);
      
      // Gradient check
      BOOST_CHECK(check_grad(cg, y, 0));  // Check gradient w.r.t. parameter 0
    }

3. **Test different batch sizes** and input dimensions
4. **Verify GPU/CPU consistency** if applicable
5. **Test error conditions** and exception handling

Testing Checklist
~~~~~~~~~~~~~~~~~~

Before submitting your contribution:

- [ ] All existing tests pass (``make test``)
- [ ] New functionality has corresponding tests
- [ ] Tests cover edge cases and error conditions
- [ ] Gradient checks pass for differentiable operations
- [ ] Performance benchmarks show no significant regressions
- [ ] Memory usage is reasonable (no obvious leaks)
- [ ] Code works on both CPU and GPU (if applicable)

5. Creating Pull Requests
-------------------------

Preparing Your Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Ensure your branch is up to date**::

    git fetch upstream
    git rebase upstream/main

2. **Run the full test suite**::

    make test

3. **Clean up your commit history**::

    # Interactive rebase to squash/reorder commits
    git rebase -i upstream/main

4. **Write clear commit messages**::

    Add efficient matrix multiplication operation
    
    - Implement optimized CPU and GPU kernels
    - Add comprehensive unit tests with gradient checks
    - Include benchmark comparisons showing 2x speedup
    - Update documentation with usage examples
    
    Fixes #123

Pull Request Description
~~~~~~~~~~~~~~~~~~~~~~~~

Your PR description should include:

**Summary**
  Brief description of what the PR accomplishes

**Changes Made**
  * Bullet point list of specific changes
  * Reference any related issues

**Testing**
  * Description of tests added or modified
  * Performance impact (if applicable)
  * Platforms tested on

**Checklist**
  - [ ] Tests pass locally
  - [ ] Code follows style guidelines
  - [ ] Documentation updated (if needed)
  - [ ] No breaking changes (or clearly documented)

Example PR Template::

    ## Summary
    This PR adds support for efficient sparse matrix operations in ED-Batch.

    ## Changes Made
    - Added `SparseMatrixMultiply` operation in `nodes-matrixmultiply.cc`
    - Implemented CPU and GPU kernels with optimized memory access patterns
    - Added comprehensive unit tests in `test-nodes.cc`
    - Updated documentation with usage examples

    Fixes #456

    ## Testing
    - All existing tests pass
    - Added 15 new unit tests covering various sparse matrix configurations
    - Benchmark shows 3x speedup on sparse matrices with <10% density
    - Tested on Ubuntu 20.04 with GCC 9.4 and CUDA 11.1

    ## Performance Impact
    - No impact on dense matrix operations
    - Significant speedup for sparse operations (see benchmarks in PR)
    - Memory usage reduced by ~50% for sparse matrices

    ## Checklist
    - [x] Tests pass locally (`make test`)
    - [x] Code follows ED-Batch style guidelines
    - [x] Documentation updated
    - [x] No breaking changes to existing API

6. Code Review Process
----------------------

Review Guidelines
~~~~~~~~~~~~~~~~~

**For Contributors**

When your PR is under review:

* **Respond promptly** to reviewer feedback
* **Ask questions** if feedback is unclear
* **Make requested changes** in separate commits (don't squash during review)
* **Test thoroughly** after making changes
* **Be patient** - thorough reviews take time

**For Reviewers**

When reviewing PRs:

* **Check correctness** of the implementation
* **Verify test coverage** is adequate
* **Ensure code style** follows project standards
* **Consider performance implications**
* **Test the changes** locally when possible
* **Provide constructive feedback** with specific suggestions

Review Checklist
~~~~~~~~~~~~~~~~~

**Functionality**
- [ ] Code solves the stated problem
- [ ] Implementation is correct and efficient
- [ ] Edge cases are handled properly
- [ ] Error conditions are handled gracefully

**Code Quality**
- [ ] Code follows ED-Batch style guidelines
- [ ] Functions are well-documented
- [ ] Variable names are clear and descriptive
- [ ] No obvious performance issues

**Testing**
- [ ] Adequate test coverage for new functionality
- [ ] Tests are well-structured and maintainable
- [ ] All tests pass
- [ ] No regressions in existing functionality

**Documentation**
- [ ] Code changes are documented
- [ ] API documentation is updated (if applicable)
- [ ] Examples are provided for new features

Common Review Comments
~~~~~~~~~~~~~~~~~~~~~~

**Performance Issues**::

    // Avoid: Creating temporary objects in hot paths
    Tensor temp = some_operation(input);
    return another_operation(temp);
    
    // Prefer: In-place operations when possible
    some_operation_inplace(input, output);

**Memory Management**::

    // Avoid: Manual memory allocation
    float* data = new float[size];
    
    // Prefer: Use ED-Batch's memory management
    Tensor result(Dim({rows, cols}));

**Error Handling**::

    // Avoid: Silent failures
    if (input.d.size() == 0) return;
    
    // Prefer: Clear error messages
    DYNET_INVALID_ARG("Input tensor cannot be empty in operation X");

7. Merge Procedures
-------------------

Merge Criteria
~~~~~~~~~~~~~~

A pull request is ready for merge when:

* **All tests pass** on continuous integration
* **Code review is complete** with approvals from maintainers
* **Documentation is updated** as needed
* **No merge conflicts** with the main branch
* **Performance regressions** are addressed or justified

Merge Process
~~~~~~~~~~~~~

1. **Final review** by project maintainers
2. **Squash and merge** for feature additions
3. **Regular merge** for bug fixes (to preserve history)
4. **Update changelog** for significant changes
5. **Close related issues** automatically via commit messages

Post-Merge Activities
~~~~~~~~~~~~~~~~~~~~~

After your PR is merged:

* **Monitor for issues** related to your changes
* **Respond to bug reports** if they arise
* **Consider follow-up improvements** based on feedback
* **Update documentation** if usage patterns emerge

Troubleshooting Common Issues
-----------------------------

Build Issues
~~~~~~~~~~~~

**Eigen not found**::

    # Ensure EIGEN3_INCLUDE_DIR is set correctly
    export EIGEN3_INCLUDE_DIR=/path/to/eigen-3.4.0

**CUDA compilation errors**::

    # Check CUDA version compatibility
    nvcc --version
    # ED-Batch is tested with CUDA 11.1

**Linking errors**::

    # Ensure LD_LIBRARY_PATH includes ED-Batch libraries
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/lib

Test Failures
~~~~~~~~~~~~~

**Gradient check failures**::

    # Often indicates incorrect backward implementation
    # Ensure dEdxi accumulates (uses +=, not =)

**Numerical precision issues**::

    # Use appropriate tolerances in tests
    BOOST_CHECK_CLOSE(expected, actual, 1e-5);  // 1e-5 relative tolerance

**Memory-related failures**::

    # Check for uninitialized memory access
    # Ensure fx is completely set in forward pass

Performance Issues
~~~~~~~~~~~~~~~~~~

**Slow compilation**::

    # Use fewer parallel jobs if memory is limited
    make -j2  # instead of -j4

**Runtime performance**::

    # Profile with appropriate tools
    # Check for unnecessary memory allocations
    # Verify batch sizes are appropriate

Getting Help
------------

If you encounter issues not covered in this guide:

1. **Search existing issues** on GitHub
2. **Check the documentation** in ``doc/source/``
3. **Ask questions** in GitHub discussions or issues
4. **Provide detailed information** when asking for help:
   
   * System configuration
   * Complete error messages
   * Minimal reproducible example
   * Steps you've already tried

Contributing to Documentation
-----------------------------

Documentation improvements are always welcome:

* **Fix typos** and grammatical errors
* **Add examples** for complex features
* **Improve clarity** of existing explanations
* **Add missing documentation** for new features

Documentation follows reStructuredText format and is built with Sphinx.

Conclusion
----------

Contributing to ED-Batch helps advance the state of efficient dynamic neural network computation. This walkthrough provides the foundation for successful contributions, but don't hesitate to ask questions or seek clarification when needed.

Remember:

* **Start small** with your first contributions
* **Follow the established patterns** in the codebase
* **Test thoroughly** before submitting
* **Be patient** with the review process
* **Learn from feedback** to improve future contributions

Thank you for contributing to ED-Batch!

.. seealso::
   
   * :ref:`code_style` - Detailed coding standards and conventions
   * :ref:`contributing` - General contribution guidelines
   * :ref:`debugging` - Debugging tips and techniques
