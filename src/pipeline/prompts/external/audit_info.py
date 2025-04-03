from llama_index.core.prompts import PromptTemplate


AUDIT_PROMPT_EXTERNAL = PromptTemplate(
    "Please analyze the following firearm details based on available sources and general firearm knowledge. "
    "Audit the information to determine if it is likely accurate or contains errors. "
    "- If all the information is likely accurate, respond with 'True'.\n"
    "- If any key information is likely inaccurate, respond with 'False' and explain briefly.\n"
    "Focus on serial number accuracy, manufacturing details, and historical records. Avoid false positives and only flag issues if you are highly confident an error exists.\n"
    "\n---\n"
    "### **Example Queries and Responses:**\n"
    "\n#### **Example 1:**\n"
    "**Query:** Audit the firearm description stating that serial number RH201355 belongs to a Tula SKS 1955/56 manufactured in 1955.\n"
    "**Answer:** True. The serial number RH201355 follows the 1955 transitional Tula SKS dating system, where the receiver cover date was replaced with a Cyrillic-letter serial number marking. This matches known Tula Arsenal production data for 1955 SKS rifles.\n"
    "\n#### **Example 2:**\n"
    "**Query:** Audit the firearm description stating that the serial number CDB95 corresponds to a Cugir Model 56 SKS produced in 1959.\n"
    "**Answer:** False. The serial number CDB95 likely corresponds to the 1958 production batch of Cugir Model 56 SKS rifles, not 1959. Romanian SKS rifles followed an inconsistent prefix system, but records indicate that this specific prefix aligns with 1958 production at the Cugir Arsenal.\n"
    "\n---\n"
    "**Now, using the information below, audit the provided firearm specifications.**\n"
    "------\n"
    "{context_str}\n"
    "------\n"
    "**Specifications:** {query_str}\n"
    "**Answer:**"
)
