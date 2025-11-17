"""
Prompt templates for RAG generation
"""

from typing import Optional


class PromptTemplate:
    """Prompt template manager"""
    
    @staticmethod
    def rag_prompt(context: str, question: str, instructions: Optional[str] = None) -> str:
        """
        Generate RAG prompt
        
        Args:
            context: Retrieved context
            question: User question
            instructions: Optional additional instructions
            
        Returns:
            Formatted prompt
        """
        base_instructions = "Use the following context to answer the question accurately and concisely."
        
        if instructions:
            base_instructions = instructions
        
        prompt = f"""{base_instructions}

Context:
{context}

Question: {question}

Answer:"""
        return prompt
    
    @staticmethod
    def summarization_prompt(text: str, max_length: Optional[int] = None) -> str:
        """
        Generate summarization prompt
        
        Args:
            text: Text to summarize
            max_length: Optional maximum length hint
            
        Returns:
            Formatted prompt
        """
        length_hint = f" in {max_length} sentences" if max_length else ""
        prompt = f"Summarize the following text{length_hint}:\n\n{text}\n\nSummary:"
        return prompt
    
    @staticmethod
    def question_answering_prompt(context: str, question: str) -> str:
        """
        Generate question answering prompt
        
        Args:
            context: Context information
            question: Question to answer
            
        Returns:
            Formatted prompt
        """
        return PromptTemplate.rag_prompt(
            context=context,
            question=question,
            instructions="Answer the question based on the provided context. If the context doesn't contain enough information, say so."
        )
    
    @staticmethod
    def extraction_prompt(context: str, extraction_type: str) -> str:
        """
        Generate information extraction prompt
        
        Args:
            context: Context to extract from
            extraction_type: Type of information to extract (e.g., "key points", "dates", "names")
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Extract {extraction_type} from the following context:

Context:
{context}

{extraction_type.capitalize()}:"""
        return prompt

