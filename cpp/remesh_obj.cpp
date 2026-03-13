#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>
#include <CGAL/IO/OBJ.h>
#include <iostream>
#include <fstream>
#include <string>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;
namespace PMP = CGAL::Polygon_mesh_processing;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input.obj> <output.obj> <target_edge_length>\n";
        return 1;
    }

    std::string const input_path{argv[1]};
    std::string const output_path{argv[2]};
    double const target_edge_length = std::stod(argv[3]);

    Mesh mesh;
    if (!CGAL::IO::read_OBJ(input_path, mesh)) {
        std::cerr << "Error: failed to read OBJ from " << input_path << "\n";
        return 1;
    }

    std::cout << "Input: " << mesh.number_of_vertices() << " vertices, "
              << mesh.number_of_faces() << " faces\n";

    PMP::isotropic_remeshing(
        faces(mesh),
        target_edge_length,
        mesh,
        CGAL::parameters::number_of_iterations(3)
    );

    PMP::smooth_shape(
        faces(mesh),
        mesh,
        0.01, // time step
        CGAL::parameters::number_of_iterations(10)
    );

    std::cout << "Output: " << mesh.number_of_vertices() << " vertices, "
              << mesh.number_of_faces() << " faces\n";

    std::ofstream out(output_path);
    if (!out || !CGAL::IO::write_OBJ(out, mesh)) {
        std::cerr << "Error: failed to write OBJ to " << output_path << "\n";
        return 1;
    }

    return 0;
}
