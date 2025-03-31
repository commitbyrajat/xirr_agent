from dotenv import load_dotenv

from wealth.wealth_agent import fxn_invoke

if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")
    # query = "Use available tools to calculate arc cosine of 0.5."
    # invoke(query)

    query = """
    I have made the following investments and received returns on different dates:  
    - Invested ₹10,000 on January 1, 2020  
    - Received ₹2,000 on January 1, 2021  
    - Received ₹4,000 on January 1, 2022  
    - Received ₹6,000 on January 1, 2023  
    - Received ₹8,000 on January 1, 2024  
    
    ### Instructions:
    1. **Analyze the method signatures and descriptions** of the available tools before invoking them.  
    2. Calculate **XNPV** before calculating **XIRR**.  
    3. Use `parse_date` to convert the date from a string.  
    4. Use only the available tools to calculate **XIRR** and **XNPV**.  
    5. Do **not** generate code to perform the calculations; strictly use the tools.  
    6. Convert **XIRR** to percentage format before returning the result.  
    7. Return the final result in the following **JSON format**:  
    
    ```json
    {
      "XNPV": <calculated_xnpv_value>,
      "XIRR": "<calculated_xirr_value_in_percentage>%"
    }
    ```
    
    Ensure that the response **only contains a valid JSON output** without any additional explanation or text.
    """
    fxn_invoke(query)
