# Contributing to RAG Starter Kit

Thank you for your interest in contributing to the RAG Starter Kit! This project aims to provide a production-ready RAG platform with enterprise-grade features. We welcome contributions from the community.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to Contribute

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes, new features, or improvements
- **Documentation**: Improve or add documentation
- **Testing**: Add test cases or improve test coverage
- **Performance**: Optimize performance and scalability

## Getting Started

### Development Setup

1. **Fork the repository** and clone your fork:

   ```bash
   git clone https://github.com/your-username/rag-starter-kit.git
   cd rag-starter-kit
   ```

2. **Set up the development environment**:

   ```bash
   # Copy development environment variables
   cp .env.development .env
   
   # Start the development environment
   docker-compose up -d
   ```

3. **Install development dependencies**:

   ```bash
   # Backend dependencies
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   
   # Frontend dependencies
   cd ../frontend
   npm install
   ```

4. **Install pre-commit hooks**:

   ```bash
   cd backend
   pre-commit install
   ```

### Running Tests

Before submitting any changes, ensure all tests pass:

```bash
# Backend tests
cd backend
pytest --cov=app --cov-report=html

# Frontend tests
cd frontend
npm test
npm run test:coverage
```

### Code Quality

We maintain high code quality standards:

```bash
# Backend formatting and linting
cd backend
black .                 # Format code
isort .                 # Sort imports
flake8                  # Lint code
mypy .                  # Type checking

# Frontend linting
cd frontend
npm run lint           # ESLint
npm run type-check     # TypeScript checking
```

## Contribution Guidelines

### Branch Naming

Use descriptive branch names:

- `feat/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements
- `perf/description` - Performance improvements

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `ci`: CI/CD changes

Examples:

```
feat(search): add semantic search capability
fix(api): resolve document upload validation error
docs(readme): update installation instructions
```

### Pull Request Process

1. **Create a feature branch** from `main`:

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Test thoroughly**:
   - Add tests for new functionality
   - Ensure existing tests pass
   - Test manually in the development environment

4. **Update documentation** if needed:
   - Update README.md for new features
   - Add/update API documentation
   - Update inline code comments

5. **Submit a pull request**:
   - Use the pull request template
   - Provide a clear description of changes
   - Link any related issues
   - Request review from maintainers

### Pull Request Template

When creating a pull request, include:

- **Summary**: Brief description of changes
- **Type of Change**: Bug fix, new feature, breaking change, etc.
- **Testing**: How the changes were tested
- **Checklist**: Ensure all requirements are met
- **Related Issues**: Link to any related issues

## Development Workflow

### Adding New Features

1. **Check existing issues** to avoid duplicate work
2. **Create an issue** to discuss the feature first
3. **Design consideration**: Consider impact on:
   - API endpoints
   - Database schema
   - Frontend UI
   - Performance
   - Security

### Vector Database Providers

To add a new vector database provider:

1. Create provider class in `backend/app/services/vectordb/providers/`
2. Implement the `VectorDBInterface` abstract methods
3. Add provider to factory in `backend/app/services/vectordb/factory.py`
4. Update configuration in `backend/app/core/config.py`
5. Add comprehensive tests
6. Update documentation

### Frontend Components

When adding React components:

1. Follow existing component patterns
2. Use TypeScript with proper type definitions
3. Implement responsive design with Tailwind CSS
4. Add proper error handling
5. Include loading states for async operations
6. Add unit tests with React Testing Library

### API Endpoints

When adding new API endpoints:

1. Follow FastAPI best practices
2. Use proper Pydantic models for request/response
3. Implement proper error handling
4. Add comprehensive documentation
5. Include rate limiting if needed
6. Add integration tests

## Architecture Guidelines

### Backend Architecture

- **Service Pattern**: Business logic in service classes
- **Repository Pattern**: Data access through repositories
- **Dependency Injection**: Use FastAPI's dependency system
- **Async/Await**: Use async operations for I/O
- **Error Handling**: Comprehensive error handling with proper status codes

### Frontend Architecture

- **Component Composition**: Small, reusable components
- **State Management**: React hooks for local state
- **API Integration**: Centralized API calls
- **Error Boundaries**: Proper error handling
- **Performance**: Optimize renders and bundle size

### Database Design

- **Migrations**: Use Alembic for schema changes
- **Indexing**: Proper database indexing for performance
- **Relationships**: Clear foreign key relationships
- **Constraints**: Data integrity constraints

## Security Considerations

- **Input Validation**: Validate all user inputs
- **Authentication**: Secure API endpoints appropriately
- **Authorization**: Implement proper access controls
- **Data Privacy**: Handle sensitive data appropriately
- **Rate Limiting**: Prevent abuse with rate limiting
- **Security Headers**: Include appropriate security headers

## Performance Guidelines

- **Database Queries**: Optimize database queries
- **Caching**: Implement appropriate caching strategies
- **Async Operations**: Use async for I/O operations
- **Resource Management**: Proper resource cleanup
- **Monitoring**: Include performance metrics

## Documentation Standards

- **Code Comments**: Document complex logic
- **API Documentation**: Use FastAPI's automatic documentation
- **README Updates**: Keep README.md current
- **Architecture Docs**: Update architecture documentation
- **Examples**: Provide usage examples

## Issue Reporting

When reporting issues:

1. **Search existing issues** first
2. **Use issue templates** for bugs and features
3. **Provide reproduction steps** for bugs
4. **Include environment details**:
   - Operating system
   - Python/Node.js versions
   - Docker version
   - Browser (for frontend issues)

### Bug Reports

Include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Error messages and logs
- Environment information
- Screenshots if applicable

### Feature Requests

Include:

- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Potential impact on existing functionality

## Release Process

Maintainers follow this release process:

1. **Version Bump**: Update version numbers
2. **Changelog**: Update CHANGELOG.md
3. **Testing**: Comprehensive testing
4. **Documentation**: Update documentation
5. **Tag Release**: Create Git tag
6. **Deploy**: Deploy to staging/production

## Community Guidelines

- **Be Respectful**: Treat all community members with respect
- **Be Constructive**: Provide constructive feedback
- **Be Patient**: Understand that maintainers are volunteers
- **Help Others**: Help other contributors when possible
- **Stay On Topic**: Keep discussions relevant to the project

## Getting Help

- **Documentation**: Check the [docs/](docs/) folder
- **Issues**: Search existing issues for solutions
- **Discussions**: Use GitHub Discussions for questions
- **API Docs**: Check the interactive API documentation at `/docs`

## Recognition

Contributors will be recognized in:

- **README.md**: Listed in contributors section
- **Release Notes**: Mentioned in release notes
- **GitHub**: Contribution graphs and statistics

## Resources

- [Architecture Documentation](docs/)
- [API Reference](http://localhost:8000/docs)
- [Deployment Guide](docs/6_Deployment_and_Infrastructure.md)
- [Security Guidelines](docs/7_Security_and_Compliance.md)

Thank you for contributing to RAG Starter Kit! 🚀
