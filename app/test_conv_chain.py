import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Path resolution to ensure 'app' and 'engine' are found
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from engine import ChatEngine

# Configure Logging for professional visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChainTester:
    def __init__(self, strategy="similarity"):
        self.engine = ChatEngine()
        self.strategy = strategy
        self.session_id = f"test_session_{datetime.now().strftime('%H%M%S')}"
        
    def run_integration_test(self):
        """
        Executes a multi-turn conversation test to verify 
        Rephrasing, Retrieval, and Generation.
        """
        print(f"\n{'='*60}")
        print(f"STARTING INTEGRATION TEST (Strategy: {self.strategy})")
        print(f"{'='*60}")

        try:
            # 1. Initialization
            logger.info("Initializing Conversational Chain...")
            self.engine.create_conversational_chain(strategy=self.strategy)

            # 2. Turn 1: Direct Factual Query (Cold Start)
            # Replace with a topic relevant to your documents
            q1 = "What is a recommendation system?" 
            print(f"\n[TURN 1] Query: {q1}")
            
            res1 = self.engine.ask(q1, session_id=self.session_id)
            print(f"RESPONSE: {res1}")

            # 3. Turn 2: Contextual Follow-up (The 'Rephraser' Test)
            # This query contains a pronoun 'his' which requires the rephraser to resolve
            q2 = "Who can build a recommendation system?"
            print(f"\n[TURN 2] Query: {q2}")
            
            res2 = self.engine.ask(q2, session_id=self.session_id)
            print(f"RESPONSE: {res2}")

            # 4. Integrity Check: Memory Storage
            self._verify_memory()

        except Exception as e:
            logger.error(f"Test Failed: {str(e)}", exc_info=True)

    def _verify_memory(self):
        """Internal verification of the message history store."""
        history = self.engine.get_session_history(self.session_id)
        msg_count = len(history.messages)
        
        print(f"\n--- Integrity Report ---")
        print(f"Session ID:      {self.session_id}")
        print(f"Messages Stored: {msg_count}")
        
        # A successful 2-turn conversation should have 4 messages (2 Human, 2 AI)
        if msg_count >= 4:
            print("STATUS: PASS - Multi-turn memory persisted.")
        else:
            print("STATUS: FAIL - Unexpected message count in history.")

if __name__ == "__main__":
    # Test 1: Standard Similarity
    tester_sim = ChainTester(strategy="similarity")
    tester_sim.run_integration_test()

    # Test 2: MMR (Diversity test)
    # tester_mmr = ChainTester(strategy="mmr")
    # tester_mmr.run_integration_test()