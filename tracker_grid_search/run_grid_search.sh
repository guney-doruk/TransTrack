#!/bin/bash
# Grid Search Execution Script
# This script provides easy commands to run and manage the grid search

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRID_SEARCH_SCRIPT="${SCRIPT_DIR}/grid_search_tracker.py"
ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze_grid_search.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  run           Run the full grid search"
    echo "  dry-run       Show what would be executed without running"
    echo "  resume N      Resume grid search from experiment N"
    echo "  analyze       Analyze completed grid search results"
    echo "  status        Check status of running/completed experiments"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 run                 # Run full grid search"
    echo "  $0 dry-run             # Preview experiments"
    echo "  $0 resume 15           # Resume from experiment 15"
    echo "  $0 analyze             # Analyze results"
}

function run_grid_search() {
    echo -e "${GREEN}Starting grid search...${NC}"
    python3 "${GRID_SEARCH_SCRIPT}" "$@"
}

function run_dry_run() {
    echo -e "${YELLOW}Running dry-run (no actual execution)...${NC}"
    python3 "${GRID_SEARCH_SCRIPT}" --dry-run
}

function resume_grid_search() {
    if [ -z "$1" ]; then
        echo -e "${RED}Error: Please specify experiment number to resume from${NC}"
        echo "Usage: $0 resume N"
        exit 1
    fi
    echo -e "${GREEN}Resuming grid search from experiment $1...${NC}"
    python3 "${GRID_SEARCH_SCRIPT}" --resume-from "$1"
}

function analyze_results() {
    echo -e "${GREEN}Analyzing grid search results...${NC}"
    python3 "${ANALYZE_SCRIPT}"
}

function check_status() {
    echo -e "${GREEN}Checking grid search status...${NC}"
    
    RESULTS_FILE="/cta/users/grad4/master/TransTrack/output/latest_finetune/tracker_grid_search/grid_search_results.json"
    
    if [ ! -f "${RESULTS_FILE}" ]; then
        echo -e "${YELLOW}No grid search results found. Have you started the grid search?${NC}"
        exit 0
    fi
    
    # Count experiments
    TOTAL=$(python3 -c "import json; data=json.load(open('${RESULTS_FILE}')); print(len(data))")
    SUCCESS=$(python3 -c "import json; data=json.load(open('${RESULTS_FILE}')); print(sum(1 for r in data if r['status']=='success'))")
    FAILED=$(python3 -c "import json; data=json.load(open('${RESULTS_FILE}')); print(sum(1 for r in data if r['status']=='failed'))")
    
    echo ""
    echo "Grid Search Status:"
    echo "  Total experiments run: ${TOTAL}"
    echo "  Successful: ${SUCCESS}"
    echo "  Failed: ${FAILED}"
    echo ""
    
    # Expected total
    EXPECTED=30  # 5 weight pairs * 6 unmatch thresholds
    if [ ${TOTAL} -eq ${EXPECTED} ]; then
        echo -e "${GREEN}✓ Grid search completed!${NC}"
    else
        REMAINING=$((EXPECTED - TOTAL))
        echo -e "${YELLOW}⚠ ${REMAINING} experiments remaining${NC}"
    fi
}

# Main script logic
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

case "$1" in
    run)
        run_grid_search
        ;;
    dry-run)
        run_dry_run
        ;;
    resume)
        resume_grid_search "$2"
        ;;
    analyze)
        analyze_results
        ;;
    status)
        check_status
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Error: Unknown option '$1'${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac