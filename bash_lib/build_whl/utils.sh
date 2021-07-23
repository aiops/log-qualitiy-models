#!/bin/bash

function help {
   # Display Help
   echo "Publish model by building whl."
   echo
   echo "Syntax: build_whl$1 [-h | -t ['level' | 'ling'] <name>]"
   echo "options:"
   echo "h     Print this Help."
   echo "t     Define log quality analysis type. Possible values: 'level' | 'ling'"
   echo
   echo "This script can usually be called without arguments if the prepare script was calles before."
   echo "Otherwise, the log quality type ['level' | 'ling'] and the model name need to be defined."
   echo "This script can be used to update an existing model. ['level' | 'ling'] and the model name need to be defined."
   echo
}

function warning {
    echo "build-whl: $*" >&2
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
                tflag=true
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

    if [ ! "$#" = 1 ] && [ "$tflag" = true ]; then
        warning "error: If -t is defined, model name needs to be defined as well"\
            "=> Exiting."
        return 102
    fi

    if [ "$#" = 1 ] && [ ! "$tflag" = true ]; then
        warning "error: Model name is defined, -t falg needs to be defined as well"\
            "=> Exiting."
        return 103
    fi

    model_name=$1

    return 0
}