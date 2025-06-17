import streamlit as st
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
import os

SUBSCRIPTION_ID = os.environ['SUBSCRIPTION_ID']
RESOURCE_GROUP = os.environ['RESOURCE_GROUP']
WORKSPACE_NAME = os.environ['WORKSPACE_NAME']
DEFAULT_EXPERIMENT_NAME = "simpel-algoritme-4"
ENVIRONMENT_NAME = "newenv"
SCRIPT_NAME = "script.py"

st.title("🚀 Start een Azure ML Experiment")

# Voeg tekstveld toe voor de experimentnaam
experiment_name = st.text_input("🧪 Voer de naam van het experiment in:", value=DEFAULT_EXPERIMENT_NAME)

if st.button("Start experiment"):
    try:
        with st.spinner("🔌 Verbinden met Azure ML Workspace..."):
            ws = Workspace(
                subscription_id=SUBSCRIPTION_ID,
                resource_group=RESOURCE_GROUP,
                workspace_name=WORKSPACE_NAME
            )

        with st.spinner(f"🧪 Starten van experiment '{experiment_name}'..."):
            experiment = Experiment(workspace=ws, name=experiment_name)

            # Laad environment
            env = Environment.get(workspace=ws, name=ENVIRONMENT_NAME)

            # Configureer ScriptRun
            config = ScriptRunConfig(
                source_directory= '.',
                script=SCRIPT_NAME,
                environment=env
            )

            run = experiment.submit(config=config)

        st.info("⏳ Wachten op voltooiing...")
        run.wait_for_completion(show_output=True)

        st.success("✅ Experiment voltooid!")
        st.markdown(f"🔗 [Bekijk in Azure ML Studio]({run.get_portal_url()})")

    except Exception as e:
        st.error(f"❌ Fout bij uitvoeren: {e}")
