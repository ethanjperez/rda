Testing
=======================================================================================================================


Let's take a look at how 🤗 Transformer models are tested and how you can write new tests and improve the existing ones.

There are 2 test suites in the repository:

1. ``tests`` -- tests for the general API
2. ``examples`` -- tests primarily for various applications that aren't part of the API

How transformers are tested
-----------------------------------------------------------------------------------------------------------------------

1. Once a PR is submitted it gets tested with 9 CircleCi jobs. Every new commit to that PR gets retested. These jobs are defined in this `config file <https://github.com/huggingface/transformers/blob/master/.circleci/config.yml>`__, so that if needed you can reproduce the same environment on your machine.
   
   These CI jobs don't run ``@slow`` tests.
   
2. There are 3 jobs run by `github actions <https://github.com/huggingface/transformers/actions>`__:

   * `torch hub integration <https://github.com/huggingface/transformers/blob/master/.github/workflows/github-torch-hub.yml>`__:  checks whether torch hub integration works.

   * `self-hosted (push) <https://github.com/huggingface/transformers/blob/master/.github/workflows/self-push.yml>`__: runs fast tests on GPU only on commits on ``master``. It only runs if a commit on ``master`` has updated the code in one of the following folders: ``src``, ``tests``, ``.github`` (to prevent running on added model cards, notebooks, etc.)
     
   * `self-hosted runner <https://github.com/huggingface/transformers/blob/master/.github/workflows/self-scheduled.yml>`__: runs normal and slow tests on GPU in ``tests`` and ``examples``:

   .. code-block:: bash

    RUN_SLOW=1 pytest tests/
    RUN_SLOW=1 pytest examples/

   The results can be observed `here <https://github.com/huggingface/transformers/actions>`__.



Running tests
-----------------------------------------------------------------------------------------------------------------------





Choosing which tests to run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This document goes into many details of how tests can be run. If after reading everything, you need even more details you will find them `here <https://docs.pytest.org/en/latest/usage.html>`__.

Here are some most useful ways of running tests.

Run all:

.. code-block:: console

   pytest

or:

.. code-block:: bash

   make test

Note that the latter is defined as:

.. code-block:: bash

   python -m pytest -n auto --dist=loadfile -s -v ./tests/

which tells pytest to:

* run as many test processes as they are CPU cores (which could be too many if you don't have a ton of RAM!)
* ensure that all tests from the same file will be run by the same test process
* do not capture output
* run in verbose mode



Getting the list of all tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All tests of the test suite:

.. code-block:: bash

   pytest --collect-only -q

All tests of a given test file:

.. code-block:: bash

   pytest tests/test_optimization.py --collect-only -q


   
Run a specific test module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run an individual test module:

.. code-block:: bash

   pytest tests/test_logging.py
   

Run specific tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since unittest is used inside most of the tests, to run specific subtests you need to know the name of the unittest class containing those tests. For example, it could be:

.. code-block:: bash

   pytest tests/test_optimization.py::OptimizationTest::test_adam_w

Here:

* ``tests/test_optimization.py`` - the file with tests
* ``OptimizationTest`` - the name of the class
* ``test_adam_w`` - the name of the specific test function

If the file contains multiple classes, you can choose to run only tests of a given class. For example:

.. code-block:: bash

   pytest tests/test_optimization.py::OptimizationTest


will run all the tests inside that class.

As mentioned earlier you can see what tests are contained inside the ``OptimizationTest`` class by running:

.. code-block:: bash

   pytest tests/test_optimization.py::OptimizationTest --collect-only -q

  
You can run tests by keyword expressions.

To run only tests whose name contains ``adam``:

.. code-block:: bash

   pytest -k adam tests/test_optimization.py

To run all tests except those whose name contains ``adam``:

.. code-block:: bash

   pytest -k "not adam" tests/test_optimization.py

And you can combine the two patterns in one:


.. code-block:: bash

   pytest -k "ada and not adam" tests/test_optimization.py



Run only modified tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can run the tests related to the unstaged files or the current branch (according to Git) by using `pytest-picked <https://github.com/anapaulagomes/pytest-picked>`__. This is a great way of quickly testing your changes didn't break anything, since it won't run the tests related to files you didn't touch.

.. code-block:: bash

    pip install pytest-picked

.. code-block:: bash

    pytest --picked

All tests will be run from files and folders which are modified, but not
yet committed.

Automatically rerun failed tests on source modification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`pytest-xdist <https://github.com/pytest-dev/pytest-xdist>`__ provides a
very useful feature of detecting all failed tests, and then waiting for
you to modify files and continuously re-rerun those failing tests until
they pass while you fix them. So that you don't need to re start pytest
after you made the fix. This is repeated until all tests pass after
which again a full run is performed.

.. code-block:: bash

    pip install pytest-xdist

To enter the mode: ``pytest -f`` or ``pytest --looponfail``

File changes are detected by looking at ``looponfailroots`` root
directories and all of their contents (recursively). If the default for
this value does not work for you, you can change it in your project by
setting a configuration option in ``setup.cfg``:

.. code-block:: ini

    [tool:pytest]
    looponfailroots = transformers tests

or ``pytest.ini``/``tox.ini`` files:

.. code-block:: ini

    [pytest]
    looponfailroots = transformers tests

This would lead to only looking for file changes in the respective
directories, specified relatively to the ini-file’s directory.

`pytest-watch <https://github.com/joeyespo/pytest-watch>`__ is an
alternative implementation of this functionality.


Skip a test module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to run all test modules, except a few you can exclude them by giving an explicit list of tests to run. For example, to run all except ``test_modeling_*.py`` tests:

.. code-block:: bash

   pytest `ls -1 tests/*py | grep -v test_modeling`


Clearing state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CI builds and when isolation is important (against speed), cache should
be cleared:

.. code-block:: bash

    pytest --cache-clear tests

Running tests in parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned earlier ``make test`` runs tests in parallel via ``pytest-xdist`` plugin (``-n X`` argument, e.g. ``-n 2`` to run 2 parallel jobs).

``pytest-xdist``'s ``--dist=`` option allows one to control how the tests are grouped. ``--dist=loadfile`` puts the tests located in one file onto the same process.

Since the order of executed tests is different and unpredictable, if
running the test suite with ``pytest-xdist`` produces failures (meaning
we have some undetected coupled tests), use
`pytest-replay <https://github.com/ESSS/pytest-replay>`__ to replay the
tests in the same order, which should help with then somehow reducing
that failing sequence to a minimum.

Test order and repetition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's good to repeat the tests several times, in sequence, randomly, or
in sets, to detect any potential inter-dependency and state-related bugs
(tear down). And the straightforward multiple repetition is just good to
detect some problems that get uncovered by randomness of DL.


Repeat tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `pytest-flakefinder <https://github.com/dropbox/pytest-flakefinder>`__:

.. code-block:: bash

   pip install pytest-flakefinder

And then run every test multiple times (50 by default):

.. code-block:: bash

   pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
   
.. note::
   This plugin doesn't work with ``-n`` flag from ``pytest-xdist``.
   
.. note::
   There is another plugin ``pytest-repeat``, but it doesn't work with ``unittest``.


Run tests in a random order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install pytest-random-order

Important: the presence of ``pytest-random-order`` will automatically
randomize tests, no configuration change or command line options is
required.

As explained earlier this allows detection of coupled tests - where one
test's state affects the state of another. When ``pytest-random-order``
is installed it will print the random seed it used for that session,
e.g:

.. code-block:: bash

   pytest tests
   [...]
   Using --random-order-bucket=module
   Using --random-order-seed=573663

So that if the given particular sequence fails, you can reproduce it by
adding that exact seed, e.g.:

.. code-block:: bash

   pytest --random-order-seed=573663
   [...]
   Using --random-order-bucket=module
   Using --random-order-seed=573663

It will only reproduce the exact order if you use the exact same list of
tests (or no list at all). Once you start to manually narrowing
down the list you can no longer rely on the seed, but have to list them
manually in the exact order they failed and tell pytest to not randomize
them instead using ``--random-order-bucket=none``, e.g.:

.. code-block:: bash

   pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py

To disable the shuffling for all tests:

.. code-block:: bash

    pytest --random-order-bucket=none

By default ``--random-order-bucket=module`` is implied, which will
shuffle the files on the module levels. It can also shuffle on
``class``, ``package``, ``global`` and ``none`` levels. For the complete
details please see its `documentation <https://github.com/jbasko/pytest-random-order>`__.

Another randomization alternative is: ``pytest-randomly`` <https://github.com/pytest-dev/pytest-randomly>`__. This module has a very similar functionality/interface, but it doesn't have the bucket modes available in ``pytest-random-order``. It has the same problem of imposing itself once installed.

Look and feel variations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pytest-sugar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`pytest-sugar <https://github.com/Frozenball/pytest-sugar>`__ is a
plugin that improves the look-n-feel, adds a progressbar, and show tests
that fail and the assert instantly. It gets activated automatically upon
installation.

.. code-block:: bash
                
   pip install pytest-sugar

To run tests without it, run:

.. code-block:: bash

    pytest -p no:sugar

or uninstall it.



Report each sub-test name and its progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a single or a group of tests via ``pytest`` (after
``pip install pytest-pspec``):

.. code-block:: bash

   pytest --pspec tests/test_optimization.py 



Instantly shows failed tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`pytest-instafail <https://github.com/pytest-dev/pytest-instafail>`__
shows failures and errors instantly instead of waiting until the end of
test session.

.. code-block:: bash

    pip install pytest-instafail

.. code-block:: bash

    pytest --instafail

To GPU or not to GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a GPU-enabled setup, to test in CPU-only mode add ``CUDA_VISIBLE_DEVICES=""``:

.. code-block:: bash
                
    CUDA_VISIBLE_DEVICES="" pytest tests/test_logging.py

or if you have multiple gpus, you can specify which one is to be used by ``pytest``. For example, to use only the second gpu if you have gpus ``0`` and ``1``, you can run:

.. code-block:: bash
                
    CUDA_VISIBLE_DEVICES="1" pytest tests/test_logging.py

This is handy when you want to run different tasks on different GPUs.
    
And we have these decorators that require the condition described by the marker.

``
@require_torch
@require_tf
@require_multigpu
@require_non_multigpu
@require_torch_tpu
@require_torch_and_cuda
``

Some decorators like ``@parametrized`` rewrite test names, therefore ``@require_*`` skip decorators have to be listed last for them to work correctly. Here is an example of the correct usage:

.. code-block:: python

    @parameterized.expand(...)
    @require_multigpu
    def test_integration_foo():
    
There is no problem whatsoever with ``@pytest.mark.parametrize`` (but it only works with non-unittests) - can use it in any order.

This section will be expanded soon once our work in progress on those decorators is finished.

Inside tests:

* How many GPUs are available:

.. code-block:: bash

   torch.cuda.device_count()


   


Output capture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During test execution any output sent to ``stdout`` and ``stderr`` is
captured. If a test or a setup method fails, its according captured
output will usually be shown along with the failure traceback.

To disable output capturing and to get the ``stdout`` and ``stderr``
normally, use ``-s`` or ``--capture=no``:

.. code-block:: bash

   pytest -s tests/test_logging.py

To send test results to JUnit format output:

.. code-block:: bash

   py.test tests --junitxml=result.xml


Color control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To have no color (e.g., yellow on white background is not readable):

.. code-block:: bash

   pytest --color=no tests/test_logging.py



Sending test report to online pastebin service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a URL for each test failure:

.. code-block:: bash

   pytest --pastebin=failed tests/test_logging.py

This will submit test run information to a remote Paste service and
provide a URL for each failure. You may select tests as usual or add for
example -x if you only want to send one particular failure.

Creating a URL for a whole test session log:

.. code-block:: bash

   pytest --pastebin=all tests/test_logging.py



Writing tests
-----------------------------------------------------------------------------------------------------------------------

🤗 transformers tests are based on ``unittest``, but run by ``pytest``, so most of the time features from both systems can be used.

You can read `here <https://docs.pytest.org/en/stable/unittest.html>`__ which features are supported, but the important thing to remember is that most ``pytest`` fixtures don't work. Neither parametrization, but we use the module ``parameterized`` that works in a similar way.


Parametrization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, there is a need to run the same test multiple times, but with different arguments. It could be done from within the test, but then there is no way of running that test for just one set of arguments.

.. code-block:: python
                
    # test_this1.py
    import unittest
    from parameterized import parameterized
    class TestMathUnitTest(unittest.TestCase):
        @parameterized.expand([
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ])
        def test_floor(self, name, input, expected):
            assert_equal(math.floor(input), expected)

Now, by default this test will be run 3 times, each time with the last 3 arguments of ``test_floor`` being assigned the corresponding arguments in the parameter list.

and you could run just the ``negative`` and ``integer`` sets of params with:

.. code-block:: bash

   pytest -k "negative and integer" tests/test_mytest.py

or all but ``negative`` sub-tests, with:

.. code-block:: bash

   pytest -k "not negative" tests/test_mytest.py

Besides using the ``-k`` filter that was just mentioned, you can find out the exact name of each sub-test and run any or all of them using their exact names. 
        
.. code-block:: bash
                
    pytest test_this1.py --collect-only -q

and it will list:
                
.. code-block:: bash

    test_this1.py::TestMathUnitTest::test_floor_0_negative
    test_this1.py::TestMathUnitTest::test_floor_1_integer
    test_this1.py::TestMathUnitTest::test_floor_2_large_fraction

So now you can run just 2 specific sub-tests:

.. code-block:: bash

    pytest test_this1.py::TestMathUnitTest::test_floor_0_negative  test_this1.py::TestMathUnitTest::test_floor_1_integer
   
The module `parameterized <https://pypi.org/project/parameterized/>`__ which is already in the developer dependencies of ``transformers`` works for both: ``unittests`` and ``pytest`` tests.

If, however, the test is not a ``unittest``, you may use ``pytest.mark.parametrize`` (or you may see it being used in some existing tests, mostly under ``examples``).

Here is the same example, this time using ``pytest``'s ``parametrize`` marker:

.. code-block:: python

    # test_this2.py
    import pytest
    @pytest.mark.parametrize(
        "name, input, expected",
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ],
    )
    def test_floor(name, input, expected):
        assert_equal(math.floor(input), expected)

Same as with ``parameterized``, with ``pytest.mark.parametrize`` you can have a fine control over which sub-tests are run, if the ``-k`` filter doesn't do the job. Except, this parametrization function creates a slightly different set of names for the sub-tests. Here is what they look like:
        
.. code-block:: bash
                
    pytest test_this2.py --collect-only -q

and it will list:
                
.. code-block:: bash

    test_this2.py::test_floor[integer-1-1.0]
    test_this2.py::test_floor[negative--1.5--2.0]
    test_this2.py::test_floor[large fraction-1.6-1]       

So now you can run just the specific test:

.. code-block:: bash

    pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]

as in the previous example.

    

Temporary files and directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using unique temporary files and directories are essential for parallel test running, so that the tests won't overwrite each other's data. Also we want to get the temp files and directories removed at the end of each test that created them. Therefore, using packages like ``tempfile``, which address these needs is essential.

However, when debugging tests, you need to be able to see what goes into the temp file or directory and you want to know it's exact path and not having it randomized on every test re-run.

A helper class :obj:`transformers.test_utils.TestCasePlus` is best used for such purposes. It's a sub-class of :obj:`unittest.TestCase`, so we can easily inherit from it in the test modules.

Here is an example of its usage:

.. code-block:: python

    from transformers.testing_utils import TestCasePlus
    class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()

This code creates a unique temporary directory, and sets :obj:`tmp_dir` to its location.

In this and all the following scenarios the temporary directory will be auto-removed at the end of test, unless ``after=False`` is passed to the helper function.

* Create a temporary directory of my choice and delete it at the end - useful for debugging when you want to monitor a specific directory:

.. code-block:: python

    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir(tmp_dir="./tmp/run/test")

* Create a temporary directory of my choice and do not delete it at the end---useful for when you want to look at the temp results:

.. code-block:: python

    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir(tmp_dir="./tmp/run/test", after=False)

* Create a temporary directory of my choice and ensure to delete it right away---useful for when you disabled deletion in the previous test run and want to make sure the that temporary directory is empty before the new test is run:

.. code-block:: python

   def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir(tmp_dir="./tmp/run/test", before=True)

.. note::
   In order to run the equivalent of ``rm -r`` safely, only subdirs of the project repository checkout are allowed if an explicit obj:`tmp_dir` is used, so that by mistake no ``/tmp`` or similar important part of the filesystem will get nuked. i.e. please always pass paths that start with ``./``.

.. note::
   Each test can register multiple temporary directories and they all will get auto-removed, unless requested otherwise.


Skipping tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is useful when a bug is found and a new test is written, yet the
bug is not fixed yet. In order to be able to commit it to the main
repository we need make sure it's skipped during ``make test``.

Methods:

-  A **skip** means that you expect your test to pass only if some
   conditions are met, otherwise pytest should skip running the test
   altogether. Common examples are skipping windows-only tests on
   non-windows platforms, or skipping tests that depend on an external
   resource which is not available at the moment (for example a
   database).

-  A **xfail** means that you expect a test to fail for some reason. A
   common example is a test for a feature not yet implemented, or a bug
   not yet fixed. When a test passes despite being expected to fail
   (marked with pytest.mark.xfail), it’s an xpass and will be reported
   in the test summary.

One of the important differences between the two is that ``skip``
doesn't run the test, and ``xfail`` does. So if the code that's buggy
causes some bad state that will affect other tests, do not use
``xfail``.

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Here is how to skip whole test unconditionally:

.. code-block:: python

    @unittest.skip("this bug needs to be fixed")
    def test_feature_x():

or via pytest:

.. code-block:: python

    @pytest.mark.skip(reason="this bug needs to be fixed")

or the ``xfail`` way:

.. code-block:: python

    @pytest.mark.xfail
    def test_feature_x():

Here is how to skip a test based on some internal check inside the test:

.. code-block:: python

    def test_feature_x():
        if not has_something():
            pytest.skip("unsupported configuration")

or the whole module:

.. code-block:: python

    import pytest
    if not pytest.config.getoption("--custom-flag"):
        pytest.skip("--custom-flag is missing, skipping tests", allow_module_level=True)

or the ``xfail`` way:

.. code-block:: python

    def test_feature_x():
        pytest.xfail("expected to fail until bug XYZ is fixed")

Here is how to skip all tests in a module if some import is missing:

.. code-block:: python

    docutils = pytest.importorskip("docutils", minversion="0.3")

-  Skip a test based on a condition:

.. code-block:: python

    @pytest.mark.skipif(sys.version_info < (3,6), reason="requires python3.6 or higher")
    def test_feature_x():

or:

.. code-block:: python

    @unittest.skipIf(torch_device == "cpu", "Can't do half precision")
    def test_feature_x():
   
or skip the whole module:

.. code-block:: python

    @pytest.mark.skipif(sys.platform == 'win32', reason="does not run on windows")
    class TestClass():
        def test_feature_x(self):

More details, example and ways are `here <https://docs.pytest.org/en/latest/skipping.html>`__.

Custom markers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Slow tests

Tests that are too slow (e.g. once downloading huge model files) are marked with:

.. code-block:: python

    from transformers.testing_utils import slow
    @slow
    def test_integration_foo():

To run such tests set ``RUN_SLOW=1`` env var, e.g.:

.. code-block:: bash

    RUN_SLOW=1 pytest tests
    
Some decorators like ``@parametrized`` rewrite test names, therefore ``@slow`` and the rest of the skip decorators ``@require_*`` have to be listed last for them to work correctly. Here is an example of the correct usage:

.. code-block:: python

    @parameterized.expand(...)
    @slow
    def test_integration_foo():

Testing the stdout/stderr output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to test functions that write to ``stdout`` and/or ``stderr``,
the test can access those streams using the ``pytest``'s `capsys
system <https://docs.pytest.org/en/latest/capture.html>`__. Here is how
this is accomplished:

.. code-block:: python

    import sys
    def print_to_stdout(s): print(s)
    def print_to_stderr(s): sys.stderr.write(s)
    def test_result_and_stdout(capsys):
        msg = "Hello"
        print_to_stdout(msg)
        print_to_stderr(msg)
        out, err = capsys.readouterr() # consume the captured output streams
        # optional: if you want to replay the consumed streams:
        sys.stdout.write(out)
        sys.stderr.write(err)
        # test:
        assert msg in out
        assert msg in err

And, of course, most of the time, ``stderr`` will come as a part of an
exception, so try/except has to be used in such a case:

.. code-block:: python

    def raise_exception(msg): raise ValueError(msg)
    def test_something_exception():
        msg = "Not a good value"
        error = ''
        try:
            raise_exception(msg)
        except Exception as e:
            error = str(e)
            assert msg in error, f"{msg} is in the exception:\n{error}"

Another approach to capturing stdout is via ``contextlib.redirect_stdout``:

.. code-block:: python

    from io import StringIO
    from contextlib import redirect_stdout
    def print_to_stdout(s): print(s)
    def test_result_and_stdout():
        msg = "Hello"
        buffer = StringIO()
        with redirect_stdout(buffer):
            print_to_stdout(msg)
        out = buffer.getvalue()
        # optional: if you want to replay the consumed streams:
        sys.stdout.write(out)
        # test:
        assert msg in out

An important potential issue with capturing stdout is that it may
contain ``\r`` characters that in normal ``print`` reset everything that
has been printed so far. There is no problem with ``pytest``, but with
``pytest -s`` these characters get included in the buffer, so to be able
to have the test run with and without ``-s``, you have to make an extra
cleanup to the captured output, using ``re.sub(r'~.*\r', '', buf, 0, re.M)``.

But, then we have a helper context manager wrapper to automatically take
care of it all, regardless of whether it has some ``\r``'s in it or
not, so it's a simple:

.. code-block:: python

    from transformers.testing_utils import CaptureStdout
    with CaptureStdout() as cs:
        function_that_writes_to_stdout()
    print(cs.out)

Here is a full test example:

.. code-block:: python

    from transformers.testing_utils import CaptureStdout
    msg = "Secret message\r"
    final = "Hello World"
    with CaptureStdout() as cs:
        print(msg + final)
    assert cs.out == final+"\n", f"captured: {cs.out}, expecting {final}"

If you'd like to capture ``stderr`` use the :obj:`CaptureStderr` class
instead:

.. code-block:: python

    from transformers.testing_utils import CaptureStderr
    with CaptureStderr() as cs:
        function_that_writes_to_stderr()
    print(cs.err)

If you need to capture both streams at once, use the parent
:obj:`CaptureStd` class:

.. code-block:: python

    from transformers.testing_utils import CaptureStd
    with CaptureStd() as cs:
        function_that_writes_to_stdout_and_stderr()
    print(cs.err, cs.out)



Capturing logger stream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to validate the output of a logger, you can use :obj:`CaptureLogger`:

.. code-block:: python

    from transformers import logging
    from transformers.testing_utils import CaptureLogger

    msg = "Testing 1, 2, 3"
    logging.set_verbosity_info()
    logger = logging.get_logger("transformers.tokenization_bart")
    with CaptureLogger(logger) as cl:
        logger.info(msg)
    assert cl.out, msg+"\n"


Testing with environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to test the impact of environment variables for a specific test you can use a helper decorator ``transformers.testing_utils.mockenv``

.. code-block:: python

    from transformers.testing_utils import mockenv
    class HfArgumentParserTest(unittest.TestCase):
        @mockenv(TRANSFORMERS_VERBOSITY="error")
        def test_env_override(self):
            env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)


Getting reproducible results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some situations you may want to remove randomness for your tests. To
get identical reproducable results set, you will need to fix the seed:

.. code-block:: python

    seed = 42

    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np
    np.random.seed(seed)

    # tf RNG
    tf.random.set_seed(seed)

Debugging tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start a debugger at the point of the warning, do this:

.. code-block:: bash

    pytest tests/test_logging.py -W error::UserWarning --pdb
