#!/bin/bash

export source_directory="${BASH_SOURCE[0]%prepare.sh}"
if [[ "$source_directory" == "" ]]; then
    source_directory='./'
fi

function create_model_dir {
    type=$1
    name=$2

    model_dir="$source_directory../"$type"_quality/$name"
    if [ -d "$model_dir" ]; then
        warning "error: directory $model_dir already exists"
        exit 22
    else
        warning "Creating $model_dir"
        mkdir "$model_dir"
    fi

    return model_dir
}

source "${source_directory}/utils.sh"

initialise_variables
retval=$?
if [[ retval -ne 0 ]]; then
    exit $retval
fi

process_command_arguments "$@"
retval=$?
if [[ retval -ne 0 ]]; then
    exit $retval
fi

create_model_dir $analysis_type $model_name