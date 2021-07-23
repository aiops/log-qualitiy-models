#!/bin/bash

export source_directory="${BASH_SOURCE[0]%prepare.sh}"
if [[ "$source_directory" == "" ]]; then
    source_directory='./'
fi

function create_model_dir {
    local model_dir="$1"
    if [ -d "$model_dir" ]; then
        warning "error: directory $model_dir already exists"
        exit 22
    else
        warning "Creating $model_dir"
        mkdir -p "$model_dir"
    fi
}

function copy_and_prepare_template_files {
    local name="$1"
    local model_dir="$2"
    local model_dir_dir="$3"

    warning "creating init file in $model_dir_dir"
    touch "$model_dir_dir/__init__.py"

    warning "copying template class file into $model_dir_dir"
    cp "$template_class_file" "$model_dir_dir/$name.py"

    warning "copying setup file into $model_dir"
    cp "$template_setup_file" "$model_dir/setup.py"

    name_camel_case=$(echo "$name" | sed -r 's/(^|-|_)(\w)/\U\2/g')
    warning "Transformed $name to camel case $name_camel_case"

    warning "Preparing setup file"
    sed -i "s/<NAME>/$name/" "$model_dir/setup.py"
    warning "Preparing class file"
    sed -i "s/<CLS>/$name_camel_case/" "$model_dir_dir/$name.py"
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

model_dir="$source_directory../../"$analysis_type"_quality/$model_name"
model_dir_dir="$source_directory../../"$analysis_type"_quality/$model_name/$model_name"

create_model_dir $model_dir_dir
copy_and_prepare_template_files $model_name $model_dir $model_dir_dir

echo "$(realpath $source_directory../build_whl/.tmp)"
echo "$(realpath $model_dir)" > "$source_directory../build_whl/.tmp"
echo "$(realpath $model_dir_dir)" >> "$source_directory../build_whl/.tmp"

echo "####### Preparation successful. #######"
echo "Next steps: "
echo "1.) Copy model file into ./$analysis_type""_quality/$model_name/$model_name. The file need to be names 'model'."
echo "2.) Adjust ./$analysis_type""_quality/$model_name/setup.py"
echo "3.) Adjust ./$analysis_type""_quality/$model_name/$model_name/$model_name.py"
echo "4.) Run 'build_whl'"
