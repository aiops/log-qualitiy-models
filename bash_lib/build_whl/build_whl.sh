#!/bin/bash

export source_directory="${BASH_SOURCE[0]%build_whl.sh}"
if [[ "$source_directory" == "" ]]; then
    source_directory='./'
fi
export source_directory_abs=$(realpath $source_directory)

function read_tmp {
    local tmp_path="$source_directory.tmp"

    if ! [ -f "$tmp_path" ]; then
        warning "error: preparation file $source_directory.tmp not found."
        warning "Specify the model that you want to publish."
        warning "Use -t [level | ling] and <model name> to specify."
        return 111
    fi

    model_dir=$(head -n 1 $tmp_path)
    model_dir_dir=$(tail -n 1 $tmp_path)

    return 0
}

function check_model_dirs {
    if ! [ -d $model_dir ]; then
        warning "error: model directory $model_dir does not exist."
        warning "If you want to publish a new model, run the prepare script"\
        " first and follow the instructions."
        warning "If you want to update a model, them the correctness of the arguments."
        return 112
    fi
    if ! [ -d $model_dir_dir ]; then
        warning "error: model directory $model_dir_dir does not exist."
        warning "Probably, something went wrong during the initialization."
        warning "Did you call the prepare script first? If yes, check the logs there."
        return 113
    fi
    if ! [ -f "$model_dir/setup.py" ]; then
        warning "error: $model_dir/setup.py does not exist."
        warning "Probably, something went wrong during the initialization."
        warning "Did you call the prepare script first? If yes, check the logs there."
        return 114
    fi
    if ! [ -f "$model_dir_dir/model" ]; then
        warning "error: The model file ($model_dir_dir/model) does not exist."
        warning "You need to copy your model into $model_dir_dir. The file must be named 'model'."
        return 115
    fi
    if ! [ -f "$model_dir_dir/$model_name.py" ]; then
        warning "error: The model class file $model_dir_dir/$model_name.py does not exist."
        warning "Probably, something went wrong during the initialization."
        warning "Did you call the prepare script first? If yes, check the logs there."
        return 116
    fi

    return 0
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

if [ -z "$model_name" ]; then
    read_tmp
else
    model_dir="$source_directory../../"$analysis_type"_quality/$model_name"
    model_dir_dir="$source_directory../../"$analysis_type"_quality/$model_name/$model_name"
fi
retval=$?
if [[ retval -ne 0 ]]; then
    exit $retval
fi
model_name=$(basename $model_dir_dir)

check_model_dirs
retval=$?
if [[ retval -ne 0 ]]; then
    exit $retval
fi


# Test model class. Simple functionality test.
python "$model_dir_dir/$model_name.py"
retval=$?
if [[ retval -ne 0 ]]; then
    warning "error: failed to execute functionality test for $model_name."
    warning "Try to execute $model_dir_dir/$model_name.py and fix occuring errors."
    exit $retval
fi


cd "$model_dir"
python "./setup.py" bdist_wheel
retval=$?
if [[ retval -ne 0 ]]; then
    warning "error: failed to build whl file"
    exit $retval
fi

rm "$source_directory_abs/.tmp"

warning "####### Whl build successful. #######"
warning "Next steps:"
warning "1.) Commit and push."
warning "2.) Create a pull request."
warning "2.) Ping @ alek-thunder (https://github.com/alek-thunder)"
