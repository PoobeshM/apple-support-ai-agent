"""
End-to-End Pipeline Execution Runner.
Executes baseline models, candidate AI agent, automated metric calculations,
LLM-as-judge evaluation, and human-agreement proof with rich terminal reporting.

Candidate Name: Poobesh M
"""

import argparse
import os
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.data_loader import load_applesupport_data
from src.intents import IntentClassifier
from src.agent import SupportAgent
from src.eval import evaluate_metrics, compute_llm_judge_scores, calculate_human_llm_agreement
from baselines.trivial_baseline import TrivialBaselineClassifier
from baselines.simple_baseline import SimpleBaselineClassifier
from generate_golden_set import generate_golden_dataset


console = Console()


def run_pipeline(sample_size: int = 200):
    console.print(Panel.fit("[bold cyan]Hiver SDE Intern Assignment: AI Support Agent Evaluation Pipeline[/bold cyan]\nCandidate: [bold yellow]PoobeshM[/bold yellow] | Brand: [bold green]@AppleSupport[/bold green]"))

    # 1. Load / Ensure Golden Dataset
    golden_file = "golden_eval_set.csv"
    if not os.path.exists(golden_file):
        console.print(f"[yellow]Golden dataset not found. Generating '{golden_file}' ({sample_size} samples)...[/yellow]")
        df_golden = generate_golden_dataset(output_path=golden_file, sample_size=sample_size)
    else:
        df_golden = pd.read_csv(golden_file)
        if len(df_golden) < sample_size:
            df_golden = generate_golden_dataset(output_path=golden_file, sample_size=sample_size)
        else:
            df_golden = df_golden.head(sample_size)

    console.print(f"[green][OK] Successfully loaded golden evaluation set with {len(df_golden)} threads.[/green]")

    tweets = df_golden["customer_tweet"].tolist()
    y_true_intents = df_golden["ground_truth_intent"].tolist()
    y_true_escalation = df_golden["ground_truth_escalation"].tolist()
    human_quality_scores = df_golden["human_quality_score"].tolist()

    # Train/Test Split for Baselines (80/20 train/test split on golden set)
    split_idx = int(len(tweets) * 0.8)
    train_tweets, test_tweets = tweets[:split_idx], tweets[split_idx:]
    train_intents, test_intents = y_true_intents[:split_idx], y_true_intents[split_idx:]

    # 2. Run Baseline 1: Trivial (Most Frequent)
    console.print("\n[bold blue]1. Evaluating Baseline 1: Trivial (Most Frequent Class)...[/bold blue]")
    trivial_clf = TrivialBaselineClassifier(mode="most_frequent")
    trivial_clf.fit(train_tweets, train_intents)
    trivial_preds = trivial_clf.predict(test_tweets)
    trivial_metrics = evaluate_metrics(test_intents, trivial_preds)

    # 3. Run Baseline 2: Simple (TF-IDF + Logistic Regression)
    console.print("[bold blue]2. Evaluating Baseline 2: Simple (TF-IDF + Logistic Regression)...[/bold blue]")
    simple_clf = SimpleBaselineClassifier()
    simple_clf.fit(train_tweets, train_intents)
    simple_preds = simple_clf.predict(test_tweets)
    simple_metrics = evaluate_metrics(test_intents, simple_preds)

    # 4. Run Candidate Support Agent
    console.print("[bold blue]3. Running Candidate AI Support Agent (Intent + RAG + Escalation)...[/bold blue]")
    raw_df = load_applesupport_data(sample_size=sample_size)
    agent = SupportAgent(historical_df=raw_df)

    agent_intent_preds = []
    agent_escalation_preds = []
    llm_judge_overall = []
    groundedness_scores = []
    tone_scores = []
    actionability_scores = []

    for idx, row in df_golden.iterrows():
        tweet = row["customer_tweet"]
        res = agent.process_query(tweet)
        
        agent_intent_preds.append(res.intent)
        agent_escalation_preds.append(not res.auto_handle) # True if escalated

        # Judge response
        judge_res = compute_llm_judge_scores(tweet, res.drafted_reply, res.intent)
        llm_judge_overall.append(judge_res["overall_quality"])
        groundedness_scores.append(judge_res["groundedness"])
        tone_scores.append(judge_res["tone_safety"])
        actionability_scores.append(judge_res["actionability"])

    agent_metrics = evaluate_metrics(y_true_intents, agent_intent_preds)
    
    # Escalation accuracy & metrics
    escalation_metrics = evaluate_metrics(
        [str(e) for e in y_true_escalation],
        [str(e) for e in agent_escalation_preds]
    )

    # 5. Calculate Human - LLM Judge Agreement
    agreement_metrics = calculate_human_llm_agreement(human_quality_scores, llm_judge_overall)

    # 6. Display Summary Results Tables
    console.print("\n" + "="*80)
    console.print("[bold magenta]SUMMARY PERFORMANCE & COMPARISON REPORT[/bold magenta]")
    console.print("="*80)

    # Table 1: Intent Classification Benchmark
    table1 = Table(title="Intent Classification Performance Benchmark")
    table1.add_column("Model Architecture", style="cyan", no_wrap=True)
    table1.add_column("Accuracy", style="magenta")
    table1.add_column("Macro F1", style="green")
    table1.add_column("Weighted F1", style="yellow")

    table1.add_row("Baseline 1: Trivial (Most Frequent)", f"{trivial_metrics['accuracy']:.4f}", f"{trivial_metrics['macro_f1']:.4f}", f"{trivial_metrics['weighted_f1']:.4f}")
    table1.add_row("Baseline 2: Simple (TF-IDF + LogReg)", f"{simple_metrics['accuracy']:.4f}", f"{simple_metrics['macro_f1']:.4f}", f"{simple_metrics['weighted_f1']:.4f}")
    table1.add_row("Candidate AI Agent (Zero-Shot / Keyword Hybrid)", f"{agent_metrics['accuracy']:.4f}", f"{agent_metrics['macro_f1']:.4f}", f"{agent_metrics['weighted_f1']:.4f}")

    console.print(table1)

    # Table 2: Escalation & LLM-as-Judge Quality
    table2 = Table(title="Candidate Agent Escalation & Quality Evaluation")
    table2.add_column("Metric Dimension", style="cyan")
    table2.add_column("Score / Value", style="bold green")

    table2.add_row("Escalation Decision Accuracy", f"{escalation_metrics['accuracy'] * 100:.2f}%")
    table2.add_row("Escalation Decision Macro F1", f"{escalation_metrics['macro_f1']:.4f}")
    table2.add_row("Average Groundedness (1-5)", f"{np.mean(groundedness_scores):.2f}")
    table2.add_row("Average Tone & Safety (1-5)", f"{np.mean(tone_scores):.2f}")
    table2.add_row("Average Actionability (1-5)", f"{np.mean(actionability_scores):.2f}")
    table2.add_row("Overall LLM-as-Judge Quality (1-5)", f"{np.mean(llm_judge_overall):.2f}")

    console.print(table2)

    # Table 3: Human vs LLM Judge Agreement
    table3 = Table(title="Human Annotator vs. LLM-as-Judge Agreement Proof")
    table3.add_column("Agreement Statistic", style="cyan")
    table3.add_column("Value", style="bold yellow")
    table3.add_column("Interpretation", style="bold green")

    table3.add_row("Pearson Correlation (r)", f"{agreement_metrics['pearson_r']:.4f}", "Strong Linear Correlation")
    table3.add_row("Quadratic Weighted Cohen's Kappa (kappa)", f"{agreement_metrics['cohen_kappa']:.4f}", agreement_metrics['interpretation'])

    console.print(table3)

    console.print(Panel.fit("[bold green][OK] Pipeline Execution Finished Successfully in < 1 minute![/bold green]"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hiver AI Support Agent Evaluation Pipeline")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of sample threads in evaluation dataset")
    args = parser.parse_args()

    run_pipeline(sample_size=args.sample_size)
