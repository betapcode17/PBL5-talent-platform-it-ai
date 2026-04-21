"""
RCTNO Structured Prompting Framework
=====================================
Implements the RCTNO structure (Role, Context, Task, Negative Constraints, Output)
to create optimized, high-quality prompts for consistent AI responses.

Framework Components:
- Role (Vai trò): Specific AI identity with expertise
- Context (Ngữ cảnh): Background information explaining the task purpose
- Task (Nhiệm vụ): Clear instructions with specific limits
- Negative Constraints (Ràng buộc tiêu cực): What AI should NOT do
- Output (Đầu ra): Desired response format
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class OutputFormat(Enum):
    """Supported output formats"""
    JSON = "json"
    MARKDOWN = "markdown"
    TABLE = "table"
    PLAIN_TEXT = "plain_text"
    BULLET_LIST = "bullet_list"
    STRUCTURED = "structured"


@dataclass
class RCTNOPrompt:
    """
    RCTNO Structured Prompt Container
    
    Attributes:
        role: The specific role/identity the AI should adopt
        context: Background information and reasoning
        task: Clear instructions on what to do
        negative_constraints: List of things NOT to do
        output: Expected output format and structure
        additional_instructions: Optional extra guidelines
    """
    role: str
    context: str
    task: str
    negative_constraints: List[str]
    output: str
    additional_instructions: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = field(default_factory=list)
    
    def to_prompt_string(self, include_sections: bool = True) -> str:
        """
        Convert RCTNO components to a formatted prompt string
        
        Args:
            include_sections: If True, include section headers
            
        Returns:
            Formatted prompt string ready for LLM
        """
        parts = []
        
        if include_sections:
            parts.extend([
                "=" * 70,
                "RCTNO STRUCTURED PROMPT",
                "=" * 70,
                ""
            ])
        
        # ROLE
        parts.extend([
            "🎭 ROLE (Vai trò):",
            self.role,
            ""
        ])
        
        # CONTEXT
        parts.extend([
            "📋 CONTEXT (Ngữ cảnh):",
            self.context,
            ""
        ])
        
        # TASK
        parts.extend([
            "✅ TASK (Nhiệm vụ):",
            self.task,
            ""
        ])
        
        # NEGATIVE CONSTRAINTS
        if self.negative_constraints:
            parts.extend([
                "❌ NEGATIVE CONSTRAINTS (Ràng buộc tiêu cực):",
                "Do NOT:"
            ])
            for constraint in self.negative_constraints:
                parts.append(f"  • {constraint}")
            parts.append("")
        
        # EXAMPLES (if provided)
        if self.examples:
            parts.extend([
                "📝 EXAMPLES:",
            ])
            for i, example in enumerate(self.examples, 1):
                parts.append(f"\nExample {i}:")
                for key, value in example.items():
                    parts.append(f"  {key}: {value}")
            parts.append("")
        
        # OUTPUT FORMAT
        parts.extend([
            "📤 OUTPUT (Đầu ra):",
            self.output,
            ""
        ])
        
        # ADDITIONAL INSTRUCTIONS
        if self.additional_instructions:
            parts.extend([
                "ℹ️ ADDITIONAL INSTRUCTIONS:",
                self.additional_instructions,
                ""
            ])
        
        return "\n".join(parts)
    
    def to_compact_prompt(self) -> str:
        """
        Compact version without section headers - for direct use with LLMs
        
        Returns:
            Minimal formatted prompt string
        """
        parts = []
        
        parts.append(self.role)
        parts.append("")
        parts.append(self.context)
        parts.append("")
        parts.append(self.task)
        parts.append("")
        
        if self.negative_constraints:
            parts.append("Do NOT:")
            for constraint in self.negative_constraints:
                parts.append(f"  • {constraint}")
            parts.append("")
        
        if self.examples:
            parts.append("Examples:")
            for i, example in enumerate(self.examples, 1):
                parts.append(f"\nExample {i}:")
                for key, value in example.items():
                    parts.append(f"  {key}: {value}")
            parts.append("")
        
        parts.append(self.output)
        
        if self.additional_instructions:
            parts.append("")
            parts.append(self.additional_instructions)
        
        return "\n".join(parts)


class RCTNOBuilder:
    """Builder class for constructing RCTNO prompts with fluent API"""
    
    def __init__(self):
        self._role: Optional[str] = None
        self._context: Optional[str] = None
        self._task: Optional[str] = None
        self._negative_constraints: List[str] = []
        self._output: Optional[str] = None
        self._additional_instructions: Optional[str] = None
        self._examples: List[Dict[str, str]] = []
    
    def set_role(self, role: str) -> 'RCTNOBuilder':
        """Set the AI role/identity"""
        self._role = role
        return self
    
    def set_context(self, context: str) -> 'RCTNOBuilder':
        """Set the background context"""
        self._context = context
        return self
    
    def set_task(self, task: str) -> 'RCTNOBuilder':
        """Set the task instructions"""
        self._task = task
        return self
    
    def add_constraint(self, constraint: str) -> 'RCTNOBuilder':
        """Add a negative constraint"""
        self._negative_constraints.append(constraint)
        return self
    
    def add_constraints(self, constraints: List[str]) -> 'RCTNOBuilder':
        """Add multiple negative constraints"""
        self._negative_constraints.extend(constraints)
        return self
    
    def set_output(self, output_format: str, 
                   structure: Optional[str] = None) -> 'RCTNOBuilder':
        """
        Set the output format and structure
        
        Args:
            output_format: Format type (json, markdown, table, etc.)
            structure: Detailed structure specification
        """
        if structure:
            self._output = f"Format: {output_format}\nStructure:\n{structure}"
        else:
            self._output = f"Format: {output_format}"
        return self
    
    def set_additional_instructions(self, instructions: str) -> 'RCTNOBuilder':
        """Set additional instructions"""
        self._additional_instructions = instructions
        return self
    
    def add_example(self, input_text: str, output_text: str) -> 'RCTNOBuilder':
        """Add an example of input/output"""
        self._examples.append({"Input": input_text, "Output": output_text})
        return self
    
    def add_examples(self, examples: List[Dict[str, str]]) -> 'RCTNOBuilder':
        """Add multiple examples"""
        self._examples.extend(examples)
        return self
    
    def build(self) -> RCTNOPrompt:
        """Build the RCTNO prompt"""
        if not self._role:
            raise ValueError("Role is required")
        if not self._context:
            raise ValueError("Context is required")
        if not self._task:
            raise ValueError("Task is required")
        if not self._output:
            raise ValueError("Output format is required")
        
        return RCTNOPrompt(
            role=self._role,
            context=self._context,
            task=self._task,
            negative_constraints=self._negative_constraints,
            output=self._output,
            additional_instructions=self._additional_instructions,
            examples=self._examples
        )


# ============================================================================
# DOMAIN-SPECIFIC TEMPLATES FOR CV SCREENING & JOB MATCHING
# ============================================================================

class CVScreeningRCTNO:
    """Pre-built RCTNO templates for CV screening"""
    
    @staticmethod
    def cv_analysis() -> RCTNOPrompt:
        """Template for analyzing CV content"""
        return (
            RCTNOBuilder()
            .set_role(
                "You are an experienced HR recruiter and career expert "
                "with 15+ years of experience in talent acquisition and CV assessment."
            )
            .set_context(
                "We are screening candidates for various positions. "
                "Your job is to thoroughly analyze CVs to extract key information "
                "and provide actionable insights about candidate qualifications."
            )
            .set_task(
                "Analyze the provided CV and extract: "
                "1. Key skills and competencies (technical and soft skills) "
                "2. Years of relevant experience "
                "3. Education and certifications "
                "4. Career progression and achievements "
                "5. Potential red flags or gaps\n"
                "Limit analysis to 5-7 key points per category. "
                "Focus on factual information only."
            )
            .add_constraints([
                "Make assumptions about candidate's potential beyond stated experience",
                "Infer missing information or read between the lines",
                "Make subjective judgments about personality or soft skills",
                "Recommend hiring/rejection decisions",
                "Include information that isn't explicitly stated in the CV"
            ])
            .set_output(
                "json",
                structure="{\n"
                "  'skills': [...],\n"
                "  'experience_years': <number>,\n"
                "  'education': [...],\n"
                "  'achievements': [...],\n"
                "  'red_flags': [...]\n"
                "}"
            )
            .set_additional_instructions(
                "Be precise and only extract information directly from the CV. "
                "If information is unclear, note it as 'Not clearly stated'."
            )
            .build()
        )
    
    @staticmethod
    def job_requirements_analysis() -> RCTNOPrompt:
        """Template for analyzing job requirements"""
        return (
            RCTNOBuilder()
            .set_role(
                "You are a senior talent acquisition specialist "
                "expert in job requirement analysis and candidate matching."
            )
            .set_context(
                "We need to understand job descriptions clearly to match them "
                "with candidate profiles. This helps in creating accurate matching "
                "criteria and identifying candidate strengths against job needs."
            )
            .set_task(
                "Extract and categorize job requirements from the job description: "
                "1. Must-have skills (critical for the role) "
                "2. Nice-to-have skills (beneficial but not required) "
                "3. Experience level required "
                "4. Education/certifications needed "
                "5. Key responsibilities "
                "Organize by priority level (Critical/Important/Nice-to-have). "
                "Limit to maximum 15 requirements."
            )
            .add_constraints([
                "Add requirements not mentioned in the job description",
                "Modify requirements to match specific candidates",
                "Interpret subjective terms (like 'motivated' or 'team player') as hard requirements",
                "Include company culture or benefits as job requirements",
                "Infer hidden requirements"
            ])
            .set_output(
                "json",
                structure="{\n"
                "  'critical': [...],\n"
                "  'important': [...],\n"
                "  'nice_to_have': [...],\n"
                "  'experience_level': <string>,\n"
                "  'education': [...]\n"
                "}"
            )
            .build()
        )
    
    @staticmethod
    def cv_job_matching() -> RCTNOPrompt:
        """Template for matching CV against job requirements"""
        return (
            RCTNOBuilder()
            .set_role(
                "You are a senior recruitment analyst with expertise in "
                "candidate-job matching and skills gap analysis."
            )
            .set_context(
                "We are matching candidates against job positions. "
                "Accurate matching helps in identifying the best candidates "
                "and understanding skill gaps for candidate development."
            )
            .set_task(
                "Compare the candidate's CV against job requirements and provide: "
                "1. Match score (0-100%) with reasoning "
                "2. Matched skills (which candidate has) "
                "3. Missing skills (gap analysis) "
                "4. Relevant experience alignment "
                "5. Recommendation (Strong/Good/Partial/Poor match) "
                "Use specific examples from CV to justify scores."
            )
            .add_constraints([
                "Give perfect scores based on general similarity",
                "Infer skills not explicitly mentioned",
                "Weight soft skills equally with hard technical skills",
                "Recommend hiring decisions",
                "Consider factors outside CV and job description (company culture fit, etc.)",
                "Modify scoring based on personal biases"
            ])
            .set_output(
                "json",
                structure="{\n"
                "  'match_score': <0-100>,\n"
                "  'recommendation': <'Strong'|'Good'|'Partial'|'Poor'>,\n"
                "  'matched_skills': [...],\n"
                "  'missing_skills': [...],\n"
                "  'reasoning': <string>\n"
                "}"
            )
            .set_additional_instructions(
                "Focus on objective matching criteria. "
                "Use specific evidence from the CV. "
                "Mark any assumptions clearly."
            )
            .build()
        )


class ChatbotRCTNO:
    """Pre-built RCTNO templates for chatbot interactions"""
    
    @staticmethod
    def career_counselor() -> RCTNOPrompt:
        """Template for career counseling chatbot"""
        return (
            RCTNOBuilder()
            .set_role(
                "You are an experienced career coach and counselor "
                "with 20+ years helping professionals with career decisions and development."
            )
            .set_context(
                "Users are seeking career guidance, advice on skill development, "
                "and help with career transition decisions. Your role is to provide "
                "personalized, actionable advice based on their background and goals."
            )
            .set_task(
                "Provide career advice based on user's question and background. "
                "Structure your response with: "
                "1. Summary of their situation "
                "2. 3-5 actionable recommendations "
                "3. Potential challenges and how to overcome them "
                "4. Resources or next steps "
                "Keep response concise (under 500 words) and practical."
            )
            .add_constraints([
                "Guarantee specific salary increases or job placements",
                "Recommend leaving their job without proper consideration",
                "Give generic advice without personalizing to their situation",
                "Make assumptions about their skills without evidence",
                "Promise specific career outcomes"
            ])
            .set_output(
                "markdown",
                structure="# Career Advice\n\n"
                "## Your Situation\n[summary]\n\n"
                "## Recommendations\n1. [action]\n2. [action]\n\n"
                "## Next Steps\n[resources]"
            )
            .build()
        )




def create_rctno_prompt(
    role: str,
    context: str,
    task: str,
    negative_constraints: Optional[List[str]] = None,
    output_format: str = "plain_text",
    examples: Optional[List[Dict[str, str]]] = None,
    additional_instructions: Optional[str] = None
) -> RCTNOPrompt:
    """
    Quick helper function to create RCTNO prompts
    
    Args:
        role: AI role/identity
        context: Background context
        task: Task instructions
        negative_constraints: List of things NOT to do
        output_format: Desired output format
        examples: Optional examples
        additional_instructions: Optional extra guidelines
        
    Returns:
        RCTNOPrompt object
    """
    builder = (
        RCTNOBuilder()
        .set_role(role)
        .set_context(context)
        .set_task(task)
        .set_output(output_format)
    )
    
    if negative_constraints:
        builder.add_constraints(negative_constraints)
    
    if examples:
        builder.add_examples(examples)
    
    if additional_instructions:
        builder.set_additional_instructions(additional_instructions)
    
    return builder.build()


def combine_rctno_with_user_input(
    rctno_prompt: RCTNOPrompt,
    user_input: str,
    prefix: str = "Now, apply the above framework to:\n\n"
) -> str:
    """
    Combine RCTNO framework with user input for complete prompt
    
    Args:
        rctno_prompt: RCTNOPrompt object
        user_input: User's specific request/data
        prefix: Text to introduce user input
        
    Returns:
        Complete prompt string
    """
    return rctno_prompt.to_compact_prompt() + "\n" + prefix + user_input
