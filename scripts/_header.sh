#!/bin/bash

print_header() {
    local service="$1"
    clear
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "      Pablo Garcia JC - CONTENEDOR      "
    echo "               ${service^^}"
    echo "╚══════════════════════════════════════╝"
    echo ""
}
