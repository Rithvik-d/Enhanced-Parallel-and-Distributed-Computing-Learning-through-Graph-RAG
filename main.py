"""
Main entry point for CDER GraphRAG system.
"""

import sys
import argparse
from pathlib import Path

from src.chatbot import CDERChatbot
from src.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CDER GraphRAG Chatbot - Hybrid RAG system for PDC education"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='.env',
        help='Path to .env file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['interactive', 'query', 'compare', 'all-modes'],
        default='interactive',
        help='Operation mode (all-modes: compare all 4 retrieval approaches)'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Query to process (for query/compare modes)'
    )
    parser.add_argument(
        '--retrieval',
        type=str,
        choices=['no-rag', 'vector-only', 'graph-only', 'hybrid'],
        default='hybrid',
        help='Retrieval mode (for query mode)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize chatbot
        chatbot = CDERChatbot(config_path=args.config, env_path=args.env)
        
        if args.mode == 'interactive':
            # Interactive chat mode
            chatbot.interactive_chat()
        
        elif args.mode == 'query':
            # Single query mode
            if not args.query:
                print("Error: --query required for query mode")
                sys.exit(1)
            
            result = chatbot.process_query(args.query, retrieval_mode=args.retrieval)
            formatted = chatbot.format_response(result)
            print(formatted)
        
        elif args.mode == 'compare':
            # Comparison mode
            if not args.query:
                print("Error: --query required for compare mode")
                sys.exit(1)
            
            results = chatbot.compare_all_retrievals(args.query)
            chatbot._display_comparison(results)
        
        elif args.mode == 'all-modes':
            # All 4 modes comparison with detailed output
            if not args.query:
                print("Error: --query required for all-modes")
                sys.exit(1)
            
            # Import and run comparison script
            from compare_all_modes import main as compare_main
            import sys as sys_module
            sys_module.argv = ['compare_all_modes.py', args.query]
            compare_main()
        
        # Cleanup
        chatbot.close()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

