#!/bin/bash

function warning {
    echo "model-publish-prepare: $*" >&2
}
export -f warning

function verbose {
    if [[ $opt_verbose = 1 ]]; then
        warning "$@"
    fi
}
export -f verbose

function initialise_variables {
    set -f

    export LC_CTYPE=C
    export LANG=C

    export opt_verbose=0

    export analysis_type=""
    export model_name=""
}

function help {
   # Display Help
   echo "Preparing directories and files for a new model that should be published."
   echo
   echo "Syntax: prepare$1 [-h | <name>]"
   echo "options:"
   echo "h     Print this Help."
   echo
   echo "Use a reasonable name. Convention: <arbitrary name>_<embedding name>_<model type>."
   echo "E.g. qulog_em_svc"
   echo
}

function process_command_arguments {
    local OPTIND
    while getopts "t:h" opt; do
        case $opt in
            h)
                help "-$analysis_type"
                exit 0
            ;;
            t)
                warning "-t Quality analysis type: $OPTARG"
                if [ $OPTARG != "level" ] && [ $OPTARG != "ling" ]; then
                    warning "error: Quality analysis type must be either 'level' or 'ling'." 
                    exit 11
                else
                    analysis_type="$OPTARG"
                fi
            ;;
            \?)
                warning "Invalid option: -$OPTARG"
                return 100
            ;;
            :)
                warning "Option -$OPTARG requires an argument."
                return 101
            ;;
        esac
    done

    shift $((OPTIND-1))

    if ! [[ "$#" = 1 ]]; then
        warning "error: Need exactly one argument =>"\
            "model name => Exiting."
        return 102
    fi

    model_name=$1

    return 0
}