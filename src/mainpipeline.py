#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Pipeline Script - End-to-End Document Processing
Combines OCR processing and LLM extraction into a single pipeline.
Processes documents from raw files to final evaluation results.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add the current directory (src) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from llm.degree_evaluator import DegreeEvaluator
    from workflow.ai_workflow import LLMOrchestrator
    from workflow.ocr_workflow import OCRWorkflow
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Error type: {type(e)}")
    print(f"   Error details: {e.__class__.__name__}: {e}")
    import traceback

    traceback.print_exc()
    print("   Make sure you're running from the backend directory")
    sys.exit(1)


class DocumentPipeline:
    """Main pipeline class that combines OCR and LLM processing."""

    def __init__(self):
        """Initialize the pipeline components."""
        self.ocr_workflow = None
        self.orchestrator = None
        self.degree_evaluator = None

    def initialize_services(self) -> bool:
        """Initialize all required services."""
        print("🔄 Initializing services...")

        # Initialize OCR workflow
        try:
            self.ocr_workflow = OCRWorkflow(
                samples_dir="samples",
                output_dir="processedData",
                language="auto",
                use_finnish_detection=True,
            )
            print("✅ OCR workflow initialized")
        except Exception as e:
            print(f"❌ Failed to initialize OCR workflow: {e}")
            return False

        # Initialize LLM orchestrator
        try:
            self.orchestrator = LLMOrchestrator()
            if not self.orchestrator.is_available():
                print("❌ LLM orchestrator not available")
                print("   Make sure GEMINI_API_KEY is set in your environment")
                return False
            print("✅ LLM orchestrator initialized")
        except Exception as e:
            print(f"❌ Failed to initialize LLM orchestrator: {e}")
            return False

        # Initialize degree evaluator
        try:
            self.degree_evaluator = DegreeEvaluator()
            print("✅ Degree evaluator initialized")
        except Exception as e:
            print(f"❌ Failed to initialize degree evaluator: {e}")
            return False

        return True

    def list_sample_files(self) -> list:
        """List all available sample files."""
        sample_dir = Path(os.path.join(os.path.dirname(__file__), "..", "samples"))
        files = []

        if sample_dir.exists():
            for file_path in sample_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in [
                    ".pdf",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tiff",
                    ".bmp",
                    ".docx",
                ]:
                    files.append(str(file_path))

        return sorted(files)

    def clean_ocr_text(self, text: str) -> str:
        """Clean OCR text for LLM processing."""
        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = [
            line.strip() for line in lines if line.strip() and len(line.strip()) > 2
        ]
        return "\n".join(cleaned_lines)

    def select_degree_program(self) -> str:
        """Let user select a degree program."""
        supported_degrees = self.degree_evaluator.get_supported_degree_programs()

        print("\n🎓 SELECT DEGREE PROGRAM:")
        for i, degree in enumerate(supported_degrees, 1):
            print(f"   {i}. {degree}")
        print(f"   {len(supported_degrees) + 1}. Custom degree")

        while True:
            try:
                choice = input(
                    f"\nEnter choice (1-{len(supported_degrees) + 1}): "
                ).strip()
                choice_num = int(choice)

                if 1 <= choice_num <= len(supported_degrees):
                    selected_degree = supported_degrees[choice_num - 1]
                    print(f"✅ Selected: {selected_degree}")
                    return selected_degree
                elif choice_num == len(supported_degrees) + 1:
                    custom_degree = input("Enter custom degree: ").strip()
                    if custom_degree:
                        return custom_degree
                    else:
                        print("❌ Please enter a valid degree name")
                else:
                    print(f"❌ Please enter 1-{len(supported_degrees) + 1}")
            except ValueError:
                print("❌ Please enter a valid number")

    def get_student_email(self) -> str:
        """Get student email from user input."""
        print("\n📧 STUDENT EMAIL:")
        print("   Enter the student's OAMK email address (@students.oamk.fi)")

        while True:
            email = input("Enter student email: ").strip()

            # Basic email validation
            if not email:
                print("❌ Email cannot be empty")
                continue

            if "@" not in email or "." not in email:
                print("❌ Please enter a valid email address")
                continue

            # Validate OAMK student email domain
            if not email.lower().endswith("@students.oamk.fi"):
                print("❌ Email must end with @students.oamk.fi")
                print("   Example: firstname.lastname@students.oamk.fi")
                continue

            # Confirm email
            confirm = input(f"Confirm email '{email}' (y/n): ").strip().lower()
            if confirm in ["y", "yes"]:
                print(f"✅ Email confirmed: {email}")
                return email
            else:
                print("🔄 Please enter the email again")

    def get_training_type(self) -> str:
        """Get the student's requested training type (general or professional)."""
        print("\n📝 TRAINING TYPE APPLICATION:")
        print("   Are you applying for General Training or Professional Training?")
        print("   1. General Training (transferable skills, not degree-specific)")
        print("   2. Professional Training (degree-related, specialized work)")

        while True:
            choice = input("Enter choice (1 for General, 2 for Professional): ").strip()
            if choice == "1":
                print("✅ Selected: General Training")
                return "general"
            elif choice == "2":
                print("✅ Selected: Professional Training")
                return "professional"
            else:
                print("❌ Please enter 1 or 2.")

    def process_document(
        self,
        file_path: str,
        student_degree: str,
        student_email: str = None,
        training_type: str = None,
    ) -> Dict[str, Any]:
        """Process a document through the complete pipeline."""
        results = {
            "success": False,
            "file_path": os.path.relpath(
                file_path, os.path.join(os.path.dirname(__file__), "..")
            ),
            "student_degree": student_degree,
            "student_email": student_email,
            "requested_training_type": training_type,
            "processing_time": 0,
            "ocr_results": {},
            "llm_results": {},
            "error": None,
        }

        start_time = datetime.now()

        try:
            # Step 1: OCR Processing
            print("\n📄 Step 1: OCR Processing")
            print(f"   File: {os.path.basename(file_path)}")

            # Process single document using OCR workflow
            file_path_obj = Path(file_path)
            ocr_result = self.ocr_workflow.process_document(file_path_obj)

            results["ocr_results"] = {
                "success": ocr_result["success"],
                "engine": "Tesseract",
                "confidence": ocr_result.get("confidence", 0),
                "processing_time": ocr_result.get("processing_time", 0),
                "text_length": ocr_result.get("text_length", 0),
                "detected_language": ocr_result.get("detected_language", "unknown"),
                "finnish_chars_count": ocr_result.get("finnish_chars_count", 0),
            }

            if not ocr_result["success"] or not ocr_result.get("extracted_text"):
                results["error"] = ocr_result.get(
                    "error", "OCR processing failed or no text extracted"
                )
                return results

            extracted_text = ocr_result["extracted_text"]
            print(f"   ✅ OCR completed: {len(extracted_text)} characters")
            if ocr_result.get("detected_language"):
                print(f"   🌐 Detected language: {ocr_result['detected_language']}")
            if ocr_result.get("finnish_chars_count", 0) > 0:
                print(f"   🇫🇮 Finnish characters: {ocr_result['finnish_chars_count']}")

            # Save OCR text to organized directory
            ocr_output_path = self.save_ocr_text(extracted_text, file_path)
            if ocr_output_path:
                print(f"   💾 OCR text saved to: {ocr_output_path}")

            # Step 2: Text Cleaning
            print("\n🧹 Step 2: Text Cleaning")
            cleaned_text = self.clean_ocr_text(extracted_text)
            print(f"   ✅ Text cleaned: {len(cleaned_text)} characters")

            # Step 3: LLM Processing
            print("\n🤖 Step 3: LLM Processing")
            print(f"   Degree: {student_degree}")
            print(f"   Requested Training Type: {training_type}")

            llm_results = self.orchestrator.process_work_certificate(
                cleaned_text, student_degree, training_type
            )
            results["llm_results"] = llm_results

            if llm_results.get("success", False):
                print("   ✅ LLM processing completed")
                results["success"] = True
            else:
                results["error"] = llm_results.get("error", "LLM processing failed")
                print("   ❌ LLM processing failed")

        except Exception as e:
            results["error"] = str(e)
            print(f"   ❌ Pipeline error: {e}")

        finally:
            end_time = datetime.now()
            results["processing_time"] = (end_time - start_time).total_seconds()

            # Clean up results for cleaner JSON output
            results = self.clean_results_for_output(results)

        return results

    def clean_results_for_output(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up results for cleaner JSON output."""
        # Create a copy to avoid modifying the original
        cleaned_results = results.copy()

        # Clean up LLM results
        if "llm_results" in cleaned_results:
            # Keep raw_response fields for manual verification and accuracy checking
            # Raw responses are valuable for debugging and quality assurance
            pass

        return cleaned_results

    def save_ocr_text(self, text: str, file_path: str) -> str:
        """Save OCR text to organized output directory."""
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "processedData", base_name
            )
            os.makedirs(output_dir, exist_ok=True)

            output_filename = f"ocr_output_{base_name}.txt"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            return output_path
        except Exception as e:
            logger.error(f"Error saving OCR text: {e}")
            return ""

    def save_pipeline_results(self, results: Dict[str, Any]) -> str:
        """Save complete pipeline results to organized output directory."""
        try:
            base_name = os.path.splitext(os.path.basename(results["file_path"]))[0]
            output_dir = os.path.join(
                os.path.dirname(__file__), "..", "processedData", base_name
            )
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"aiworkflow_output_{base_name}_{timestamp}.json"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            return output_path
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return ""

    def display_pipeline_results(self, results: Dict[str, Any]):
        """Display complete pipeline results."""
        print(f"\n{'=' * 80}")
        print("🚀 PIPELINE RESULTS")
        print(f"📄 Document: {os.path.basename(results['file_path'])}")
        print(f"🎓 Degree: {results['student_degree']}")
        if results.get("student_email"):
            print(f"📧 Student Email: {results['student_email']}")
        print(f"{'=' * 80}")

        # Overall status
        if not results.get("success", False):
            print("❌ Pipeline failed")
            print(f"   Error: {results.get('error', 'Unknown error')}")
            return

        print("✅ Pipeline completed successfully")
        print(
            f"⏱️  Total processing time: {results.get('processing_time', 0):.2f} seconds"
        )

        # OCR Results
        ocr_results = results.get("ocr_results", {})
        if ocr_results:
            print("\n📄 OCR RESULTS:")
            print(f"   • Engine: {ocr_results.get('engine', 'N/A')}")
            print(f"   • Confidence: {ocr_results.get('confidence', 0):.1f}%")
            print(f"   • Processing time: {ocr_results.get('processing_time', 0):.2f}s")
            print(f"   • Text length: {ocr_results.get('text_length', 0)} characters")
            if ocr_results.get("detected_language"):
                print(f"   • Language: {ocr_results.get('detected_language')}")
            if ocr_results.get("finnish_chars_count", 0) > 0:
                print(
                    f"   • Finnish characters: {ocr_results.get('finnish_chars_count')}"
                )

        # LLM Results
        llm_results = results.get("llm_results", {})
        if llm_results.get("success"):
            print("\n🤖 LLM RESULTS:")
            print(f"   • Model: {llm_results.get('model_used', 'N/A')}")
            print(f"   • Processing time: {llm_results.get('processing_time', 0):.2f}s")

            # Show stages completed
            stages_completed = llm_results.get("stages_completed", {})
            if stages_completed:
                print(
                    f"   • Stages: {', '.join([stage for stage, completed in stages_completed.items() if completed])}"
                )

            # Show final evaluation results (use corrected results if available)
            correction_results = llm_results.get("correction_results")
            if correction_results and correction_results.get("success"):
                # Use corrected results
                data = correction_results.get("results", {}).get(
                    "evaluation_results", {}
                )
                print("\n🎓 FINAL EVALUATION (CORRECTED):")
                print(f"   • Document: {os.path.basename(results['file_path'])}")
                print(f"   • Degree: {results['student_degree']}")
                if results.get("student_email"):
                    print(f"   • Student Email: {results['student_email']}")
                print(
                    f"   • Total Working Hours: {data.get('total_working_hours', 'N/A')}"
                )
                print(
                    f"   • Requested Training Type: {data.get('requested_training_type', 'N/A')}"
                )
                # Show the actual calculated credits (before capping)
                calculated_credits = data.get("credits_calculated", "N/A")
                qualified_credits = data.get("credits_qualified", "N/A")
                calculation_breakdown = data.get("calculation_breakdown", "")

                # Check if the calculation breakdown indicates capping
                is_capped = (
                    "capped" in calculation_breakdown.lower()
                    if calculation_breakdown
                    else False
                )

                if (
                    calculated_credits != "N/A"
                    and qualified_credits != "N/A"
                    and (calculated_credits != qualified_credits or is_capped)
                ):
                    print(
                        f"   • Calculated Credits: {calculated_credits} ECTS (capped at {qualified_credits} ECTS)"
                    )
                else:
                    print(f"   • Calculated Credits: {calculated_credits} ECTS")
                print(f"   • Degree Relevance: {data.get('degree_relevance', 'N/A')}")

                # Show calculation breakdown
                calculation = data.get("calculation_breakdown", "")
                if calculation:
                    print(f"   • Calculation: {calculation}")

                # Show decision
                decision = data.get("decision", "")
                requested_training_type = data.get("requested_training_type", "")
                calculated_credits = data.get("credits_calculated", "")
                qualified_credits = data.get("credits_qualified", "")

                if decision and calculated_credits:
                    if decision == "REJECTED":
                        # If rejected, show just the decision without credit information
                        print(f"\n🎯 DECISION: {decision}")
                    else:
                        # If accepted, show the requested training type and credits
                        display_training_type = (
                            requested_training_type.upper()
                            if requested_training_type
                            else "PROFESSIONAL"
                        )
                        # For accepted training, show the qualified credits
                        display_credits = (
                            qualified_credits
                            if qualified_credits
                            else calculated_credits
                        )
                        print(
                            f"\n🎯 DECISION: {decision} (Student receives {display_credits} ECTS as {display_training_type} training)"
                        )
                elif decision:
                    print(f"\n🎯 DECISION: {decision}")

                # Show justification
                justification = data.get("justification", "")
                if justification:
                    print(f"📋 JUSTIFICATION: {justification}")

                # Show recommendation only for rejected cases
                decision = data.get("decision", "")
                recommendation = data.get("recommendation", "")
                if recommendation and decision == "REJECTED":
                    print(f"💡 RECOMMENDATION: {recommendation}")

            else:
                # Fall back to original evaluation results
                evaluation_results = llm_results.get("evaluation_results", {})
                if evaluation_results.get("success"):
                    data = evaluation_results.get("results", {})
                    print("\n🎓 FINAL EVALUATION:")
                    print(f"   • Document: {os.path.basename(results['file_path'])}")
                    print(f"   • Degree: {results['student_degree']}")
                    if results.get("student_email"):
                        print(f"   • Student Email: {results['student_email']}")
                    print(
                        f"   • Total Working Hours: {data.get('total_working_hours', 'N/A')}"
                    )
                    print(
                        f"   • Requested Training Type: {data.get('requested_training_type', 'N/A')}"
                    )
                    # Show the actual calculated credits (before capping)
                    calculated_credits = data.get("credits_calculated", "N/A")
                    qualified_credits = data.get("credits_qualified", "N/A")
                    calculation_breakdown = data.get("calculation_breakdown", "")

                    # Check if the calculation breakdown indicates capping
                    is_capped = (
                        "capped" in calculation_breakdown.lower()
                        if calculation_breakdown
                        else False
                    )

                    if (
                        calculated_credits != "N/A"
                        and qualified_credits != "N/A"
                        and (calculated_credits != qualified_credits or is_capped)
                    ):
                        print(
                            f"   • Calculated Credits: {calculated_credits} ECTS (capped at {qualified_credits} ECTS)"
                        )
                    else:
                        print(f"   • Calculated Credits: {calculated_credits} ECTS")

                    # Show calculation breakdown
                    calculation = data.get("calculation_breakdown", "")
                    if calculation:
                        print(f"   • Calculation: {calculation}")
                    print(
                        f"   • Degree Relevance: {data.get('degree_relevance', 'N/A')}"
                    )

                    # Show decision
                    decision = data.get("decision", "")
                    requested_training_type = data.get("requested_training_type", "")
                    calculated_credits = data.get("credits_calculated", "")
                    qualified_credits = data.get("credits_qualified", "")

                    if decision and calculated_credits:
                        if decision == "REJECTED":
                            # If rejected, show just the decision without credit information
                            print(f"\n🎯 DECISION: {decision}")
                        else:
                            # If accepted, show the requested training type and credits
                            display_training_type = (
                                requested_training_type.upper()
                                if requested_training_type
                                else "PROFESSIONAL"
                            )
                            # For accepted training, show the qualified credits
                            display_credits = (
                                qualified_credits
                                if qualified_credits
                                else calculated_credits
                            )
                            print(
                                f"\n🎯 DECISION: {decision} (Student receives {display_credits} ECTS as {display_training_type} training)"
                            )
                    elif decision:
                        print(f"\n🎯 DECISION: {decision}")

                    # Show justification
                    justification = data.get("justification", "")
                    if justification:
                        print(f"\n📋 JUSTIFICATION: {justification}")

                    # Show recommendation only for rejected cases
                    decision = data.get("decision", "")
                    recommendation = data.get("recommendation", "")
                    if recommendation and decision == "REJECTED":
                        print(f"💡 RECOMMENDATION: {recommendation}")

        print("=" * 80)


def main():
    """Main function for the complete document processing pipeline."""
    print("🚀 DOCUMENT PROCESSING PIPELINE")
    print("=" * 50)
    print("This pipeline combines OCR and LLM processing into a single workflow.")
    print("=" * 50)

    # Initialize pipeline
    pipeline = DocumentPipeline()
    if not pipeline.initialize_services():
        print("❌ Failed to initialize pipeline services")
        return

    # List available sample files
    sample_files = pipeline.list_sample_files()
    if not sample_files:
        print("❌ No sample files found in 'samples' directory")
        return

    print(f"\n📁 Found {len(sample_files)} sample files:")
    for i, file_path in enumerate(sample_files, 1):
        print(f"   {i}. {os.path.basename(file_path)}")

    # Get user selection
    while True:
        try:
            choice = input("\nEnter your choice (number) or 'q' to quit: ").strip()

            if choice.lower() == "q":
                print("👋 Goodbye!")
                return

            choice_num = int(choice)
            if 1 <= choice_num <= len(sample_files):
                selected_file = sample_files[choice_num - 1]
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(sample_files)}")
        except ValueError:
            print("❌ Please enter a valid number")

    # Select degree program
    student_degree = pipeline.select_degree_program()

    # Get student email
    student_email = pipeline.get_student_email()

    # Get training type
    training_type = pipeline.get_training_type()

    print(f"\n{'=' * 60}")
    print("🚀 STARTING PIPELINE")
    print(f"📄 Document: {os.path.basename(selected_file)}")
    print(f"🎓 Degree: {student_degree}")
    print(f"📧 Student Email: {student_email}")
    print(f"{'=' * 60}")

    # Process document through complete pipeline
    results = pipeline.process_document(
        selected_file, student_degree, student_email, training_type
    )

    # Display results
    pipeline.display_pipeline_results(results)

    # Save results
    if results.get("success", False):
        output_path = pipeline.save_pipeline_results(results)
        print(f"💾 Complete results saved to: {output_path}")

        # Show the organized output structure
        base_name = os.path.splitext(os.path.basename(selected_file))[0]
        output_base_dir = os.path.join(
            os.path.dirname(__file__), "..", "processedData", base_name
        )
        print(f"\n📁 Output organized in: {output_base_dir}")
        print(f"   ├── ocr_output_{base_name}.txt     (OCR text)")
        print(
            f"   └── aiworkflow_output_{base_name}_*.json     (AI workflow evaluation results)"
        )
    else:
        print("⚠️  Pipeline failed - no results to save")


if __name__ == "__main__":
    main()
