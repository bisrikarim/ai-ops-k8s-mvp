import typer, uvicorn, asyncio
from .collector import app as collector_app
from .rca import run_rca_agent
from .remediator import run_remediator_agent

cli = typer.Typer()

@cli.command()
def collector(port: int = 8080):
    """Lance l'agent collector (webhook Alertmanager)."""
    uvicorn.run(
        collector_app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )

@cli.command()
def rca():
    """Lance l'agent RCA (consomme NATS, appelle LLM, publie diagnostic)."""
    asyncio.run(run_rca_agent())

@cli.command()
def remediator():
    """Lance l'agent remediator (consomme RCA, génère patches K8s)."""
    asyncio.run(run_remediator_agent())

if __name__ == "__main__":
    cli()
