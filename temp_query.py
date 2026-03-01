from agent.tools import WorkspaceTools
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

tools = WorkspaceTools(
    root=Path("."),
    mariadb_host=os.getenv("MARIADB_HOST", "127.0.0.1"),
    mariadb_port=int(os.getenv("MARIADB_PORT", "3306")),
    mariadb_user=os.getenv("MARIADB_USER", "root"),
    mariadb_password=os.getenv("MARIADB_PASSWORD", ""),
    mariadb_database=os.getenv("MARIADB_DATABASE")
)

query = """
SELECT 
    a.estabelicimento AS health_facility, 
    COUNT(p.id_paciente) AS patient_count
FROM 
    aps a
JOIN 
    paciente_aps p ON a.id_aps = p.id_aps
GROUP BY 
    a.id_aps
ORDER BY 
    patient_count DESC
LIMIT 10;
"""

print("--- Top 10 Health Facilities by Patient Count ---")
print(tools.mariadb_query(query))
