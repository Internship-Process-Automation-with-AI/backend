"""
LLM prompt for extracting basic information from work certificates.
This prompt focuses on extracting structured data like names, dates, positions, etc.
"""

EXTRACTION_PROMPT = (
    "You are an expert document analyzer. Extract specific information from work certificates and employment documents.\n\n"
    "RESPOND WITH ONLY A VALID JSON OBJECT. NO OTHER TEXT.\n\n"
    "REQUIRED FIELDS:\n"
    "- employee_name: Full name of the person employed\n"
    "- position: Job title or role\n"
    "- employer: Company or organization name\n"
    "- start_date: Start date in YYYY-MM-DD format\n"
    "- end_date: End date in YYYY-MM-DD format (null if not specified)\n"
    "- employment_period: Duration description\n"
    '- document_language: "en" for English, "fi" for Finnish\n'
    '- confidence_level: "high", "medium", or "low"\n\n'
    "EXAMPLE DOCUMENT:\n"
    "October 7, 2012\n"
    "To:\n"
    "Dear Ms. Dominick,\n"
    "We are in receipt of your request for employment verification for Louis Peterson, as it relates to his application for a temporary employment visa. "
    "As the coordinator of our firm's international internship program since 2009, I can gladly provide you with the information you seek.\n"
    "Louis Peterson recently graduated from college with a degree in International Business. He applied and was accepted to our Global Partners Program "
    "program through our branch office in his hometown of Munich, Germany. He hopes to complete the 12-week internship program in New York and then "
    "seek permanent employment in our Germany office. The internship will commence on May 28, 2013 and end on July 20, 2013. His activities while "
    "employed in the internship program would involve training and entry-level functions in our Accounting, Finance, Investment, and Marketing departments.\n"
    "We sincerely hope that Mr. Peterson is able to obtain an employment visa and take advantage of this opportunity, which we feel will be greatly "
    "beneficial to both him and our firm. If I can be of further assistance, please contact me at (800) 555-2329 or by email at jphillips@capital.com.\n"
    "Sincerely,\n"
    "Jack Phillips\n"
    "Global Partners Internship Coordinator\n\n"
    "EXAMPLE OUTPUT:\n"
    "{\n"
    '    "employee_name": "Louis Peterson",\n'
    '    "position": "Intern in Global Partners Program",\n'
    '    "employer": "Capital",\n'
    '    "start_date": "2013-05-28",\n'
    '    "end_date": "2013-07-20",\n'
    '    "employment_period": "12-week internship",\n'
    '    "confidence_level": "high",\n'
    '    "extraction_notes": "Clear employment information found in internship letter"\n'
    "}\n\n"
    "Now extract from this document:\n\n"
    "{text}\n\n"
    "JSON:"
)
