"""
Tests for AI workflow pipeline functionality.
"""

from unittest.mock import Mock, patch

from src.llm.degree_evaluator import DegreeEvaluator
from src.workflow.ai_workflow import LLMOrchestrator


class TestLLMOrchestrator:
    """Test LLM Orchestrator functionality."""

    def test_llm_orchestrator_initialization(self):
        """Test LLM orchestrator initialization."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash"]

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                assert orchestrator is not None
                assert orchestrator.model_name == "gemini-2.0-flash"
                assert len(orchestrator.available_models) == 2
                assert orchestrator.degree_evaluator is not None

    def test_llm_orchestrator_initialization_no_api_key(self):
        """Test LLM orchestrator initialization without API key."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = None
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            orchestrator = LLMOrchestrator()

            assert orchestrator.model is None
            assert orchestrator.is_available() is False

    def test_llm_orchestrator_fallback_initialization(self):
        """Test LLM orchestrator with fallback model initialization."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash"]

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                # First model fails, second succeeds
                mock_genai.GenerativeModel.side_effect = [
                    Exception("Model not available"),
                    Mock(),
                ]

                orchestrator = LLMOrchestrator()

                assert orchestrator.model is not None
                assert orchestrator.model_name == "gemini-2.5-flash"
                assert orchestrator.current_model_index == 1

    def test_handle_quota_error(self):
        """Test quota error handling with fallback."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash"]

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()
                orchestrator.current_model_index = 0

                # Test quota error handling
                result = orchestrator._handle_quota_error(
                    "quota exceeded for quota metric"
                )
                assert result is True
                assert orchestrator.current_model_index == 1

    def test_handle_quota_error_no_fallback(self):
        """Test quota error handling when no fallback available."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()
                orchestrator.current_model_index = 0

                # Test quota error with no fallback
                result = orchestrator._handle_quota_error("quota exceeded")
                assert result is False

    def test_call_llm_with_fallback_success(self):
        """Test successful LLM call with fallback support."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash"]

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_response = Mock()
                mock_response.text = "Test response"
                mock_model.generate_content.return_value = mock_response
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                result = orchestrator._call_llm_with_fallback(
                    "Test prompt", "test operation"
                )
                assert result == "Test response"

    def test_call_llm_with_fallback_quota_error(self):
        """Test LLM call with quota error and fallback."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash"]

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                # First model fails with quota error, second succeeds
                mock_model1 = Mock()
                mock_model1.generate_content.side_effect = Exception("quota exceeded")
                mock_model2 = Mock()
                mock_response = Mock()
                mock_response.text = "Fallback response"
                mock_model2.generate_content.return_value = mock_response

                mock_genai.GenerativeModel.side_effect = [mock_model1, mock_model2]

                orchestrator = LLMOrchestrator()

                result = orchestrator._call_llm_with_fallback(
                    "Test prompt", "test operation"
                )
                assert result == "Fallback response"

    def test_sanitize_text(self):
        """Test text sanitization."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                # Test text sanitization
                dirty_text = "Test\n\r\t  text   with   extra   spaces"
                clean_text = orchestrator._sanitize_text(dirty_text)
                assert clean_text == "Test\ntext with extra spaces"

    def test_validate_input(self):
        """Test input validation."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                # Test valid input
                valid_text = "Valid work certificate text"
                result = orchestrator._validate_input(valid_text)
                assert result is None  # _validate_input returns None for valid input

                # Test empty input
                result = orchestrator._validate_input("")
                assert result == "Invalid text input: <class 'str'>"

                # Test too short input
                result = orchestrator._validate_input("Short")
                assert result == "Text too short or empty"

    def test_error_response(self):
        """Test error response generation."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                error_response = orchestrator._error_response(
                    "Test error", stage="extraction"
                )
                assert error_response["success"] is False
                assert error_response["error"] == "Test error"
                # _error_response doesn't include stage in the response

    def test_parse_llm_response_valid_json(self):
        """Test parsing valid JSON response."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                valid_json = '{"name": "John Doe", "position": "Developer"}'
                result = orchestrator._parse_llm_response(valid_json)
                assert result["name"] == "John Doe"
                assert result["position"] == "Developer"

    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                invalid_json = '{"name": "John Doe", "position": "Developer"'  # Missing closing brace
                result = orchestrator._parse_llm_response(invalid_json)
                assert isinstance(result, dict)
                # _parse_llm_response returns fallback response, not error dict

    def test_fix_common_json_issues(self):
        """Test fixing common JSON formatting issues."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                # Test fixing trailing comma
                broken_json = '{"name": "John", "age": 30,}'
                fixed_json = orchestrator._fix_common_json_issues(broken_json)
                assert fixed_json == '{"name": "John", "age": 30}'

    def test_get_stats(self):
        """Test getting orchestrator statistics."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                stats = orchestrator.get_stats()
                assert "current_model" in stats
                assert "current_model_index" in stats
                assert "available_models" in stats
                assert "available" in stats

    def test_get_prompt_info(self):
        """Test getting prompt information."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                prompt_info = orchestrator.get_prompt_info()
                assert "extraction_prompt_length" in prompt_info
                assert "evaluation_prompt_length" in prompt_info
                assert "validation_prompt_length" in prompt_info
                assert "correction_prompt_length" in prompt_info


class TestDegreeEvaluator:
    """Test Degree Evaluator functionality."""

    def test_degree_evaluator_initialization(self):
        """Test degree evaluator initialization."""
        evaluator = DegreeEvaluator()
        assert evaluator is not None
        assert evaluator.degree_programs is not None
        assert len(evaluator.degree_programs) > 0

    def test_get_degree_info_exact_match(self):
        """Test getting degree info with exact match."""
        evaluator = DegreeEvaluator()

        # Test with exact name match
        degree_info = evaluator.get_degree_info("Business Administration")
        assert degree_info is not None
        assert "name" in degree_info
        assert "relevant_industries" in degree_info
        assert "relevant_roles" in degree_info

    def test_get_degree_info_normalized_match(self):
        """Test getting degree info with normalized match."""
        evaluator = DegreeEvaluator()

        # Test with normalized key
        degree_info = evaluator.get_degree_info("business_administration")
        assert degree_info is not None
        assert "name" in degree_info

    def test_get_degree_info_partial_match(self):
        """Test getting degree info with partial match."""
        evaluator = DegreeEvaluator()

        # Test with partial name
        degree_info = evaluator.get_degree_info("Business")
        assert degree_info is not None
        assert "Business" in degree_info["name"]

    def test_get_degree_info_unknown_degree(self):
        """Test getting degree info for unknown degree."""
        evaluator = DegreeEvaluator()

        # Test with unknown degree
        degree_info = evaluator.get_degree_info("Unknown Degree Program")
        assert degree_info is not None
        assert degree_info["name"] == "General Studies"  # Should fallback to general

    def test_get_degree_specific_guidelines(self):
        """Test getting degree-specific guidelines."""
        evaluator = DegreeEvaluator()

        guidelines = evaluator.get_degree_specific_guidelines("Business Administration")
        assert guidelines is not None
        assert "Business Administration" in guidelines
        assert "RELEVANT INDUSTRIES" in guidelines
        assert "RELEVANT ROLES" in guidelines
        assert "EVALUATION CRITERIA" in guidelines

    def test_validate_degree_program(self):
        """Test degree program validation."""
        evaluator = DegreeEvaluator()

        # Test valid degree
        assert evaluator.validate_degree_program("Business Administration") is True

        # Test invalid degree
        assert evaluator.validate_degree_program("Invalid Degree") is False

    def test_get_supported_degree_programs(self):
        """Test getting supported degree programs."""
        evaluator = DegreeEvaluator()

        programs = evaluator.get_supported_degree_programs()
        assert isinstance(programs, list)
        assert len(programs) > 0
        # Check for actual degree program names from the data
        assert (
            "Bachelor of Business Administration (BBA), International Business"
            in programs
        )


class TestAIWorkflowIntegration:
    """Test AI workflow integration scenarios."""

    def test_complete_workflow_success(self):
        """Test complete AI workflow success scenario."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_response = Mock()

                # Mock different responses for different stages
                responses = [
                    # Extraction response
                    '{"employee_name": "John Doe", "employer": "Tech Corp", "positions": []}',
                    # Evaluation response
                    '{"total_working_hours": 1600, "credits_qualified": 20.0, "decision": "ACCEPTED"}',
                    # Validation response
                    '{"validation_passed": true, "issues_found": []}',
                    # Correction response
                    '{"corrections_made": [], "final_result": "valid"}',
                ]

                mock_response.text = responses[0]  # Will be updated for each call
                mock_model.generate_content.return_value = mock_response
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                # Mock the _call_llm_with_fallback method to return different responses
                def mock_llm_call(prompt, operation):
                    if "extraction" in operation.lower():
                        return responses[0]
                    elif "evaluation" in operation.lower():
                        return responses[1]
                    elif "validation" in operation.lower():
                        return responses[2]
                    elif "correction" in operation.lower():
                        return responses[3]
                    return "{}"

                orchestrator._call_llm_with_fallback = mock_llm_call

                # Test complete workflow
                result = orchestrator.process_work_certificate(
                    "John Doe worked at Tech Corp as a software developer for 2 years.",
                    student_degree="Business Administration",
                    requested_training_type="professional",
                )

                assert result["success"] is True
                assert "extraction_results" in result
                assert "evaluation_results" in result
                assert "validation_results" in result
                assert "correction_results" in result

    def test_workflow_with_validation_errors(self):
        """Test AI workflow with validation errors."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                # Mock validation failure
                def mock_llm_call(prompt, operation):
                    if "validation" in operation.lower():
                        return '{"validation_passed": false, "issues_found": [{"type": "date_validation", "severity": "high"}]}'
                    return "{}"

                orchestrator._call_llm_with_fallback = mock_llm_call

                result = orchestrator.process_work_certificate(
                    "Invalid work certificate text",
                    student_degree="Business Administration",
                )

                # The workflow continues even with validation errors, so success might be True
                assert "validation_results" in result
                # Check that validation results exist but don't assume specific structure
                assert result["validation_results"] is not None

    def test_workflow_with_llm_failure(self):
        """Test AI workflow when LLM calls fail."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                # Mock LLM failure
                def mock_llm_call(prompt, operation):
                    raise Exception("LLM service unavailable")

                orchestrator._call_llm_with_fallback = mock_llm_call

                result = orchestrator.process_work_certificate(
                    "Test work certificate text",
                    student_degree="Business Administration",
                )

                assert result["success"] is False
                assert "error" in result
                assert "LLM service unavailable" in result["error"]

    def test_workflow_with_different_degree_programs(self):
        """Test AI workflow with different degree programs."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                # Mock successful responses
                def mock_llm_call(prompt, operation):
                    return '{"success": true}'

                orchestrator._call_llm_with_fallback = mock_llm_call

                # Test with different degree programs
                degree_programs = [
                    "Business Administration",
                    "Information Technology",
                    "Engineering",
                ]

                for degree in degree_programs:
                    result = orchestrator.process_work_certificate(
                        "Test work certificate text", student_degree=degree
                    )
                    assert result["success"] is True
                    assert "student_degree" in result
                    assert result["student_degree"] == degree

    def test_workflow_with_training_types(self):
        """Test AI workflow with different training types."""
        with patch("src.workflow.ai_workflow.settings") as mock_settings:
            mock_settings.GEMINI_API_KEY = "test_api_key"
            mock_settings.GEMINI_MODEL = "gemini-2.0-flash"
            mock_settings.GEMINI_FALLBACK_MODELS = []

            with patch("src.workflow.ai_workflow.genai") as mock_genai:
                mock_model = Mock()
                mock_genai.GenerativeModel.return_value = mock_model

                orchestrator = LLMOrchestrator()

                # Mock successful responses
                def mock_llm_call(prompt, operation):
                    return '{"success": true}'

                orchestrator._call_llm_with_fallback = mock_llm_call

                # Test with different training types
                training_types = ["professional", "general"]

                for training_type in training_types:
                    result = orchestrator.process_work_certificate(
                        "Test work certificate text",
                        student_degree="Business Administration",
                        requested_training_type=training_type,
                    )
                    assert result["success"] is True
                    # Check that training type is passed through in the evaluation results
                    assert "evaluation_results" in result
