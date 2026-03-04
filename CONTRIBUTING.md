# Contributing

Thanks for contributing to this project.

## Development Setup

1. Clone the repository.
2. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies.

```powershell
pip install -r requirements.txt
```

4. Ensure Ollama is installed and pull required models.

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

5. Run the app.

```powershell
python app.py
```

## Branching and Commits

- Create a feature branch from `main`.
- Keep commits focused and small.
- Use clear commit messages in imperative form.

Examples:
- `Add source citation formatting in answers`
- `Fix PDF sidebar toggle behavior`

## Code Guidelines

- Follow existing project structure and naming style.
- Keep functions small and single-purpose.
- Avoid introducing breaking changes to API routes unless necessary.
- If you update behavior, also update documentation (`README.md`).

## Testing and Validation

Before opening a pull request:
- Start the app and verify it runs without errors.
- Check `/pdfs`, `/pdf/<filename>`, and `/ask` flow in the UI.
- Confirm at least one PDF in `data/` can be queried end-to-end.

## Pull Request Checklist

- Briefly describe the problem and your fix.
- Link related issues if available.
- Include screenshots or short notes for UI changes.
- Mention any new dependency and why it is needed.
- Confirm docs were updated when needed.

## Reporting Issues

When filing a bug, include:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error logs/traceback
- Environment details (OS, Python version, Ollama model names)

## Security Notes

- Do not commit secrets, API keys, or private documents.
- Keep large/private PDFs out of version control unless intended.
