# Contributing to gps_frames

Thank you for your interest in contributing to `gps_frames`! We welcome contributions from the community to help improve this project.

## Reporting Bugs

If you find a bug, please open an issue in the [issue tracker](https://github.com/the-aerospace-corporation/gps_frames/issues). Please include:

1.  A clear and descriptive title.
2.  A description of the problem and how to reproduce it.
3.  The version of `gps_frames` you are using.
4.  Any error messages or logs.

## Submitting Pull Requests

1.  **Fork the repository** and create your branch from `main`.
2.  **Install dependencies**:
    ```bash
    pip install -e .[test,docs,examples]
    ```
3.  **Make your changes**. Ensure your code is clear and well-commented.
4.  **Run tests**:
    This project uses `pytest`. You can run the tests using the Makefile:
    ```bash
    make test
    ```
    Or manually:
    ```bash
    pytest
    ```
    Ensure all tests pass before submitting.
5.  **Code Style**:
    *   We use **Black** for code formatting. Please run `black .` before committing.
    *   We use **Numpy** style for docstrings.
    *   Please include type hints for all functions.
6.  **Update Documentation**: If your changes affect the API or usage, please update the documentation in the `docs/` directory and/or `README.md`.
7.  **Submit a Pull Request**: describing your changes and referencing any related issues.

## Development Environment

We recommend using a virtual environment for development.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[test,docs,examples]
```

## License

By contributing, you agree that your contributions will be licensed under the [GNU AGPL v3 License](LICENSE).
