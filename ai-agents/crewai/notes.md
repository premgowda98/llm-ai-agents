## Crew AI

1. Building Blocks -> Crew, Agent, Tasks
2. Command to create crew `crewai create crew <name>` -> will prompt for Models, API_Key
3. The above command will create a below folder structure

```bash
agent_crewai/
├── knowledge
│   └── user_preference.txt
├── pyproject.toml
├── README.md
├── src
│   └── agent_crewai
│       ├── config
│       │   ├── agents.yaml
│       │   └── tasks.yaml
│       ├── crew.py
│       ├── __init__.py
│       ├── main.py
│       └── tools
│           ├── custom_tool.py
│           └── __init__.py
└── tests
```