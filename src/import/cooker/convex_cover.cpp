#include "import/cooker/convex_cover.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

#include "collision/mesh_surface.hpp"

namespace nuka::import::cooker {
namespace {

using Point = CoverPoint;
using Plane = CoverPlane;
using Part = ConvexCoverPart;

Point operator+(Point a, Point b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Point operator-(Point a, Point b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Point operator*(Point a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double Dot(Point a, Point b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Point Cross(Point a, Point b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double Length(Point p) { return std::sqrt(Dot(p, p)); }
Point Unit(Point p) { const double n = Length(p); return n > 0.0 ? p * (1.0 / n) : Point{}; }
Point From(math::Vec3 p) { return {p.x, p.y, p.z}; }
math::Vec3 To(Point p) { return {float(p.x), float(p.y), float(p.z)}; }
Point Min(Point a, Point b) { return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
Point Max(Point a, Point b) { return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
double Value(const Plane& p, Point x) { return Dot(p.normal, x) - p.offset; }

struct Unresolved : std::runtime_error {
    ConvexCoverStatus status;
    Unresolved(ConvexCoverStatus s, const char* message) : std::runtime_error(message), status(s) {}
};

void Numerical(const char* reason) { throw Unresolved(ConvexCoverStatus::NumericalFailure, reason); }
void Budget(const char* reason) { throw Unresolved(ConvexCoverStatus::BudgetExceeded, reason); }

bool Contains(Point p, const std::vector<Plane>& planes, double margin) {
    for (const auto& plane : planes) if (Value(plane, p) > margin) return false;
    return true;
}

void AddPlane(std::vector<Plane>& planes, Plane plane) {
    for (auto& existing : planes) {
        if (Length(existing.normal - plane.normal) < 1.0e-12) {
            existing.offset = std::min(existing.offset, plane.offset);
            return;
        }
    }
    planes.push_back(plane);
}

void AddDirection(std::vector<Point>& directions, Point normal) {
    normal = Unit(normal);
    if (!(Length(normal) > 0.0)) return;
    for (const auto& existing : directions)
        if (Length(existing - normal) < 1.0e-10) return;
    directions.push_back(normal);
}

Point Center(const std::vector<Point>& vertices) {
    Point center{};
    for (const auto& vertex : vertices) center = center + vertex;
    return center * (1.0 / double(vertices.size()));
}

std::pair<Point, Point> Bounds(const std::vector<Point>& vertices) {
    Point lower{DBL_MAX, DBL_MAX, DBL_MAX}, upper{-DBL_MAX, -DBL_MAX, -DBL_MAX};
    for (const auto& vertex : vertices) { lower = Min(lower, vertex); upper = Max(upper, vertex); }
    return {lower, upper};
}

// Vertices are intersections of three independent boundary planes.
void RebuildPart(Part& part, double tolerance) {
    std::vector<Point> vertices;
    for (size_t a = 0; a < part.planes.size(); ++a) {
        for (size_t b = a + 1; b < part.planes.size(); ++b) {
            for (size_t c = b + 1; c < part.planes.size(); ++c) {
                const auto& pa = part.planes[a];
                const auto& pb = part.planes[b];
                const auto& pc = part.planes[c];
                const Point bc = Cross(pb.normal, pc.normal);
                const double det = Dot(pa.normal, bc);
                if (std::fabs(det) < 64.0 * DBL_EPSILON) continue;
                const Point vertex = (bc * pa.offset + Cross(pc.normal, pa.normal) * pb.offset +
                    Cross(pa.normal, pb.normal) * pc.offset) * (1.0 / det);
                if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) ||
                    !std::isfinite(vertex.z) || !Contains(vertex, part.planes, tolerance)) continue;
                bool duplicate = false;
                for (const auto& point : vertices)
                    if (Length(point - vertex) <= tolerance * 4.0) { duplicate = true; break; }
                if (!duplicate) vertices.push_back(vertex);
            }
        }
    }
    if (vertices.size() < 4u) Numerical("halfspace intersection has no resolved volume");
    const Point center = Center(vertices);
    double interior_radius = DBL_MAX;
    for (const auto& plane : part.planes)
        interior_radius = std::min(interior_radius, -Value(plane, center));
    if (!(interior_radius > tolerance)) Numerical("halfspace intersection is numerically degenerate");
    std::vector<Plane> active;
    for (const auto& plane : part.planes) {
        std::vector<Point> face;
        for (const auto& vertex : vertices)
            if (std::fabs(Value(plane, vertex)) <= tolerance * 4.0) face.push_back(vertex);
        bool area = false;
        for (size_t i = 1; i + 1 < face.size() && !area; ++i)
            for (size_t j = i + 1; j < face.size(); ++j)
                if (Length(Cross(face[i] - face[0], face[j] - face[0])) >
                    tolerance * (Length(face[i] - face[0]) + Length(face[j] - face[0])))
                    area = true;
        if (area) active.push_back(plane);
    }
    if (active.size() < 4u) Numerical("halfspace intersection has insufficient facets");
    part.planes = std::move(active);
    part.vertices = std::move(vertices);
}

struct Polygon {
    std::vector<Point> vertices;
    Point normal;
};

Polygon Clip(const Polygon& polygon, Plane plane) {
    Polygon result{{}, polygon.normal};
    bool nonnegative = true, positive = false;
    for (const auto& vertex : polygon.vertices) {
        const double value = Value(plane, vertex);
        nonnegative = nonnegative && value >= 0.0;
        positive = positive || value > 0.0;
    }
    if (nonnegative && (positive || Dot(plane.normal, polygon.normal) < 0.0)) return result;
    for (size_t i = 0; i < polygon.vertices.size(); ++i) {
        const Point a = polygon.vertices[i], b = polygon.vertices[(i + 1u) % polygon.vertices.size()];
        const double da = Value(plane, a), db = Value(plane, b);
        if (da <= 0.0) result.vertices.push_back(a);
        if ((da <= 0.0) != (db <= 0.0))
            result.vertices.push_back(a + (b - a) * (da / (da - db)));
    }
    if (result.vertices.size() < 3u) result.vertices.clear();
    return result;
}

struct Cloud {
    std::vector<Polygon> polygons;
    std::vector<Point> interior_vertices;

    std::pair<double, Point> Support(Point direction) const {
        double value = -DBL_MAX;
        Point point{};
        const auto visit = [&](Point vertex) {
            const double projection = Dot(vertex, direction);
            if (projection > value) { value = projection; point = vertex; }
        };
        for (const auto& polygon : polygons) for (const auto& vertex : polygon.vertices) visit(vertex);
        for (const auto& vertex : interior_vertices) visit(vertex);
        return {value, point};
    }

    bool Empty() const { return polygons.empty() && interior_vertices.empty(); }
};

struct WorkPart {
    Part geometry;
    Cloud cloud;
};

std::vector<std::pair<size_t, size_t>> PartEdges(const Part& part, double tolerance) {
    std::vector<std::pair<size_t, size_t>> edges;
    for (size_t a = 0u; a < part.vertices.size(); ++a) {
        for (size_t b = a + 1u; b < part.vertices.size(); ++b) {
            const Plane* first = nullptr;
            for (const auto& plane : part.planes) {
                if (std::fabs(Value(plane, part.vertices[a])) > 4.0 * tolerance ||
                    std::fabs(Value(plane, part.vertices[b])) > 4.0 * tolerance) continue;
                if (!first) { first = &plane; continue; }
                if (Length(Cross(first->normal, plane.normal)) > 64.0 * DBL_EPSILON) {
                    edges.emplace_back(a, b);
                    break;
                }
            }
        }
    }
    return edges;
}

Part CutPart(const Part& parent, const std::vector<std::pair<size_t, size_t>>& edges,
             Plane cut, double tolerance) {
    Part child{parent.planes, {}};
    AddPlane(child.planes, cut);
    const auto add = [&](Point point) {
        for (const auto& existing : child.vertices)
            if (Length(existing - point) <= tolerance * 4.0) return;
        child.vertices.push_back(point);
    };
    for (const auto& vertex : parent.vertices) if (Value(cut, vertex) <= 0.0) add(vertex);
    for (const auto& edge : edges) {
        const Point a = parent.vertices[edge.first], b = parent.vertices[edge.second];
        const double da = Value(cut, a), db = Value(cut, b);
        if ((da <= 0.0) != (db <= 0.0)) add(a + (b - a) * (da / (da - db)));
    }
    return child;
}

struct Witness { Point point; collision::MeshSurfacePoint surface; };
using CellKey = std::array<uint32_t, 4>;
struct CellValue { collision::MeshSurfacePoint surface; double triangle_bound; };

class Search {
public:
    Search(collision::MeshSurfaceView source, collision::MeshSurfaceInfo info,
           const ConvexCoverParams& params, MeshQueryBackend& queries, ConvexCoverResult& result)
        : source_(source), info_(info), params_(params), queries_(queries), result_(result) {
        const auto& root = source.nodes[info.node_offset];
        lower_ = From(root.lower);
        upper_ = From(root.upper);
        const Point extent = upper_ - lower_;
        const double scale = std::max({extent.x, extent.y, extent.z});
        const double coordinates = std::max({std::fabs(lower_.x), std::fabs(lower_.y),
            std::fabs(lower_.z), std::fabs(upper_.x), std::fabs(upper_.y), std::fabs(upper_.z), scale});
        margin_ = 64.0 * FLT_EPSILON * coordinates;
        tolerance_ = 2048.0 * DBL_EPSILON * std::max(1.0, coordinates);
        epsilon_ = params.relative_error * scale;
        if (!(epsilon_ > 8.0 * margin_)) Numerical("cover tolerance is below coordinate resolution");
        std::vector<Point> centers;
        for (uint32_t i = 0; i < info.triangle_count; ++i) {
            math::Vec3 a, b, c;
            if (!collision::MeshSurfaceTriangle(source, info, i, a, b, c))
                Numerical("invalid source triangle");
            source_polygons_.push_back({{From(a), From(b), From(c)}, From((b - a).Cross(c - a))});
            centers.push_back((From(a) + From(b) + From(c)) * (1.0 / 3.0));
        }
        const auto normals = Query(centers);
        for (size_t i = 0; i < source_polygons_.size(); ++i) {
            auto& normal = source_polygons_[i].normal;
            if (Dot(normal, From(normals[i].normal)) < 0.0) normal = normal * -1.0;
        }
        for (int x = -1; x <= 1; ++x)
            for (int y = -1; y <= 1; ++y)
                for (int z = -1; z <= 1; ++z) AddDirection(stencil_, {double(x), double(y), double(z)});
    }

    std::vector<Part> Run() {
        Part root{{{{1, 0, 0}, upper_.x}, {{0, 1, 0}, upper_.y}, {{0, 0, 1}, upper_.z},
                   {{-1, 0, 0}, -lower_.x}, {{0, -1, 0}, -lower_.y}, {{0, 0, -1}, -lower_.z}}, {}};
        RebuildPart(root, tolerance_);
        std::vector<WorkPart> initial;
        initial.push_back({root, {source_polygons_, {}}});
        PopulateInterior({&initial.front()});
        auto root_cloud = std::move(initial.front().cloud);
        for (const auto& direction : stencil_)
            AddPlane(root.planes, {direction, root_cloud.Support(direction).first + margin_});
        std::deque<WorkPart> pending;
        pending.push_back({std::move(root), std::move(root_cloud)});
        std::vector<Part> accepted;
        while (!pending.empty()) {
            if (++result_.operations > params_.max_operations) Budget("cover operation budget exceeded");
            WorkPart work = std::move(pending.front());
            pending.pop_front();
            Part& part = work.geometry;
            Cloud& cloud = work.cloud;
            RebuildPart(part, tolerance_);
            if (part.planes.size() > params_.max_planes) Budget("cover plane budget exceeded");
            if (cloud.Empty()) continue;
            for (auto& plane : part.planes)
                plane.offset = std::min(plane.offset, cloud.Support(plane.normal).first + margin_);
            RebuildPart(part, tolerance_);
            Witness witness{};
            if (!FindWitness(part, witness)) { accepted.push_back(std::move(part)); continue; }
            std::vector<Point> conflict;
            Plane separation;
            if (Separate(cloud, witness, separation, conflict)) {
                AddPlane(part.planes, separation);
                pending.push_back(std::move(work));
                continue;
            }
            auto children = Split(part, cloud, witness, conflict);
            if (accepted.size() + pending.size() + children.size() > params_.max_parts)
                Budget("cover part budget exceeded");
            for (auto& child : children) pending.push_back(std::move(child));
        }
        if (accepted.empty()) Numerical("cover search returned no solid parts");
        return accepted;
    }

private:
    std::vector<collision::MeshSurfacePoint> Query(const std::vector<Point>& points) {
        std::vector<math::Vec3> positions;
        positions.reserve(points.size());
        for (const auto& point : points) positions.push_back(To(point));
        std::vector<collision::MeshSurfacePoint> hits;
        if (!positions.empty()) queries_.Query(positions, hits);
        result_.query_points += positions.size();
        if (hits.size() != positions.size()) Numerical("query backend returned an invalid result count");
        for (const auto& hit : hits)
            if (!hit.valid || !std::isfinite(hit.distance)) Numerical("unresolved source distance");
        return hits;
    }

    void PopulateInterior(const std::vector<WorkPart*>& parts) {
        std::vector<Point> vertices;
        for (const auto* part : parts)
            vertices.insert(vertices.end(), part->geometry.vertices.begin(), part->geometry.vertices.end());
        const auto hits = Query(vertices);
        std::vector<Point> probes;
        std::vector<std::pair<WorkPart*, Point>> owners;
        size_t at = 0;
        for (auto* part : parts) {
            const Point center = Center(part->geometry.vertices);
            for (const auto& vertex : part->geometry.vertices) {
                const auto& hit = hits[at++];
                if (hit.distance > margin_) continue;
                if (hit.distance < -margin_) part->cloud.interior_vertices.push_back(vertex);
                else {
                    probes.push_back(vertex + Unit(center - vertex) * (2.0 * margin_));
                    owners.push_back({part, vertex});
                }
            }
        }
        const auto probe_hits = Query(probes);
        for (size_t i = 0; i < probes.size(); ++i)
            if (probe_hits[i].distance <= 0.0f)
                owners[i].first->cloud.interior_vertices.push_back(owners[i].second);
    }

    std::pair<Point, Point> CellBounds(CellKey cell) const {
        const double divisor = std::ldexp(1.0, int(cell[0]));
        const Point size = (upper_ - lower_) * (1.0 / divisor);
        const Point center = lower_ + Point{(double(cell[1]) + 0.5) * size.x,
            (double(cell[2]) + 0.5) * size.y, (double(cell[3]) + 0.5) * size.z};
        return {center, size * 0.5};
    }

    double TriangleBound(Point center, Point extent, uint32_t triangle) const {
        math::Vec3 a, b, c;
        if (!collision::MeshSurfaceTriangle(source_, info_, triangle, a, b, c)) return DBL_MAX;
        double maximum = 0.0;
        for (uint32_t corner = 0u; corner < 8u; ++corner) {
            const Point point = center + Point{corner & 1u ? extent.x : -extent.x,
                corner & 2u ? extent.y : -extent.y, corner & 4u ? extent.z : -extent.z};
            const auto hit = collision::ClosestTrianglePoint(To(point), a, b, c);
            maximum = std::max(maximum, Length(point - From(hit.point)));
        }
        return maximum;
    }

    bool FindWitness(const Part& part, Witness& witness) {
        const auto vertex_hits = Query(part.vertices);
        double largest = epsilon_;
        bool found = false;
        for (size_t i = 0; i < part.vertices.size(); ++i)
            if (vertex_hits[i].distance > largest) {
                largest = vertex_hits[i].distance;
                witness = {part.vertices[i], vertex_hits[i]};
                found = true;
            }
        if (found) return true;
        std::vector<CellKey> cells{{0u, 0u, 0u, 0u}};
        while (!cells.empty()) {
            std::vector<CellKey> active, missing;
            std::vector<Point> positions;
            for (const auto& cell : cells) {
                const auto bounds = CellBounds(cell);
                const Point center = bounds.first, extent = bounds.second;
                bool intersects = true;
                for (const auto& plane : part.planes) {
                    const auto n = plane.normal;
                    if (Value(plane, center) - std::fabs(n.x) * extent.x -
                        std::fabs(n.y) * extent.y - std::fabs(n.z) * extent.z > margin_) {
                        intersects = false;
                        break;
                    }
                }
                if (!intersects) continue;
                active.push_back(cell);
                if (cells_.find(cell) == cells_.end()) {
                    missing.push_back(cell);
                    positions.push_back(center);
                }
            }
            if (cells_.size() + missing.size() > params_.max_cells) Budget("cover distance cell budget exceeded");
            const auto hits = Query(positions);
            for (size_t i = 0; i < missing.size(); ++i) {
                const auto bounds = CellBounds(missing[i]);
                cells_.emplace(missing[i], CellValue{hits[i],
                    TriangleBound(bounds.first, bounds.second, hits[i].triangle)});
            }
            result_.distance_cells = static_cast<uint32_t>(cells_.size());
            cells.clear();
            for (const auto& cell : active) {
                const auto bounds = CellBounds(cell);
                const auto& value = cells_.at(cell);
                const double radius = Length(bounds.second);
                const double upper = std::min(double(value.surface.distance) + radius,
                                              value.triangle_bound) + margin_;
                if (upper <= epsilon_) continue;
                if (Contains(bounds.first, part.planes, margin_) &&
                    value.surface.distance > largest) {
                    largest = value.surface.distance;
                    witness = {bounds.first, value.surface};
                    found = true;
                }
                if (found) continue;
                if (cell[0] >= 30u || radius <= margin_ * 2.0)
                    Numerical("cover distance bound reached coordinate resolution");
                if (cells.size() > static_cast<size_t>(params_.max_cells) * 8u)
                    Budget("cover distance frontier budget exceeded");
                for (uint32_t corner = 0u; corner < 8u; ++corner)
                    cells.push_back({cell[0] + 1u, cell[1] * 2u + (corner & 1u),
                        cell[2] * 2u + ((corner >> 1u) & 1u), cell[3] * 2u + (corner >> 2u)});
            }
            if (found) return true;
        }
        return false;
    }

    static bool Solve(double matrix[3][4], size_t n, double* result) {
        for (size_t column = 0; column < n; ++column) {
            size_t pivot = column;
            for (size_t row = column + 1; row < n; ++row)
                if (std::fabs(matrix[row][column]) > std::fabs(matrix[pivot][column])) pivot = row;
            if (std::fabs(matrix[pivot][column]) < DBL_MIN) return false;
            for (size_t j = column; j <= n; ++j) std::swap(matrix[pivot][j], matrix[column][j]);
            const double value = matrix[column][column];
            for (size_t j = column; j <= n; ++j) matrix[column][j] /= value;
            for (size_t row = 0; row < n; ++row) {
                if (row == column) continue;
                const double factor = matrix[row][column];
                for (size_t j = column; j <= n; ++j) matrix[row][j] -= factor * matrix[column][j];
            }
        }
        for (size_t i = 0; i < n; ++i) result[i] = matrix[i][n];
        return true;
    }

    Point ClosestSimplex(std::vector<Point>& simplex, Point witness) const {
        double best = DBL_MAX;
        Point closest{};
        std::vector<Point> selected;
        const auto count = static_cast<uint32_t>(simplex.size());
        for (uint32_t mask = 1u; mask < (1u << count); ++mask) {
            std::vector<Point> points;
            for (uint32_t i = 0u; i < count; ++i) if (mask & (1u << i)) points.push_back(simplex[i]);
            if (points.size() > 4u) continue;
            double weights[4]{1.0, 0.0, 0.0, 0.0};
            const size_t n = points.size() - 1u;
            double matrix[3][4]{};
            for (size_t i = 0; i < n; ++i) {
                const Point a = points[i + 1u] - points[0];
                for (size_t j = 0; j < n; ++j) matrix[i][j] = Dot(a, points[j + 1u] - points[0]);
                matrix[i][n] = Dot(a, witness - points[0]);
            }
            if (n && !Solve(matrix, n, weights + 1u)) continue;
            bool positive = true;
            for (size_t i = 1u; i <= n; ++i) weights[0] -= weights[i];
            double sum = 0.0;
            for (size_t i = 0; i <= n; ++i) {
                if (weights[i] < -1.0e-12 || !std::isfinite(weights[i])) positive = false;
                weights[i] = std::max(0.0, weights[i]);
                sum += weights[i];
            }
            if (!positive || !(sum > 0.0)) continue;
            Point point{};
            for (size_t i = 0; i <= n; ++i) point = point + points[i] * (weights[i] / sum);
            const double distance = Dot(point - witness, point - witness);
            if (distance < best) {
                best = distance;
                closest = point;
                selected.clear();
                for (size_t i = 0; i <= n; ++i) if (weights[i] > 1.0e-12) selected.push_back(points[i]);
            }
        }
        if (selected.empty()) Numerical("convex support simplex failed");
        simplex = std::move(selected);
        return closest;
    }

    bool Separate(const Cloud& cloud, const Witness& witness, Plane& best,
                  std::vector<Point>& conflict) const {
        Point normal = Unit(witness.point - From(witness.surface.point));
        auto support = cloud.Support(normal);
        double best_gap = Dot(witness.point, normal) - support.first;
        bool separated = best_gap > margin_ * 4.0;
        best = {normal, support.first + margin_};
        if (best_gap >= witness.surface.distance - std::max(margin_ * 4.0, epsilon_ * 0.01))
            return separated;
        std::vector<Point> simplex{support.second};
        for (uint32_t iteration = 0u; iteration < 32u; ++iteration) {
            const Point closest = ClosestSimplex(simplex, witness.point);
            const double distance = Length(witness.point - closest);
            if (simplex.size() > 1u && witness.surface.distance - distance - margin_ > epsilon_) {
                conflict = simplex;
                return false;
            }
            if (distance <= margin_ * 4.0) return separated;
            normal = Unit(witness.point - closest);
            support = cloud.Support(normal);
            const double gap = Dot(witness.point, normal) - support.first;
            if (gap > std::max(margin_ * 4.0, best_gap)) {
                best = {normal, support.first + margin_};
                best_gap = gap;
                separated = true;
            }
            if (distance - gap <= std::max(margin_ * 4.0, epsilon_ * 0.01)) return separated;
            for (const auto& point : simplex)
                if (Length(point - support.second) <= margin_) return separated;
            simplex.push_back(support.second);
        }
        return separated;
    }

    std::vector<WorkPart> Split(const Part& part, const Cloud& cloud, const Witness& witness,
                            const std::vector<Point>& conflict) {
        math::Vec3 a, b, c;
        if (!collision::MeshSurfaceTriangle(source_, info_, witness.surface.triangle, a, b, c))
            Numerical("cover witness lost its source face");
        const Point face_normal = Unit(From((b - a).Cross(c - a)));
        const Point witness_normal = Unit(witness.point - From(witness.surface.point));
        std::vector<Point> normals{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        AddDirection(normals, face_normal);
        AddDirection(normals, witness_normal);
        AddDirection(normals, Cross(witness_normal, {1, 0, 0}));
        AddDirection(normals, Cross(witness_normal, {0, 1, 0}));
        AddDirection(normals, Cross(witness_normal, {0, 0, 1}));
        std::vector<Plane> candidates;
        const auto add_candidate = [&](Plane plane) {
            double lo = DBL_MAX, hi = -DBL_MAX;
            for (const auto& vertex : part.vertices) {
                const double projection = Dot(vertex, plane.normal);
                lo = std::min(lo, projection);
                hi = std::max(hi, projection);
            }
            if (!(plane.offset > lo + 4.0 * margin_ && plane.offset < hi - 4.0 * margin_)) return;
            for (const auto& candidate : candidates)
                if ((Length(candidate.normal - plane.normal) < 1.0e-10 &&
                     std::fabs(candidate.offset - plane.offset) < margin_) ||
                    (Length(candidate.normal + plane.normal) < 1.0e-10 &&
                     std::fabs(candidate.offset + plane.offset) < margin_)) return;
            candidates.push_back(plane);
        };
        const Point center = Center(part.vertices);
        add_candidate({face_normal, Dot(face_normal, From(a))});
        for (const auto& normal : normals) {
            add_candidate({normal, Dot(normal, witness.point)});
            add_candidate({normal, Dot(normal, center)});
        }
        for (size_t i = 0; i < conflict.size(); ++i)
            for (size_t j = i + 1u; j < conflict.size(); ++j) {
                const Point normal = Unit(conflict[j] - conflict[i]);
                if (Length(normal) > 0.0)
                    add_candidate({normal, Dot(normal, (conflict[i] + conflict[j]) * 0.5)});
            }
        if (candidates.empty()) Numerical("no resolved convex cover split");
        std::vector<Point> directions = stencil_;
        for (const auto& plane : part.planes) AddDirection(directions, plane.normal);
        AddDirection(directions, face_normal);
        AddDirection(directions, witness_normal);
        const auto bounds = Bounds(part.vertices);
        const Point width = bounds.second - bounds.first;
        std::vector<Point> probes{witness.point};
        for (uint32_t x = 0; x < 5u; ++x)
            for (uint32_t y = 0; y < 5u; ++y)
                for (uint32_t z = 0; z < 5u; ++z) {
                    const Point point = bounds.first + Point{(x + 0.5) * width.x / 5.0,
                        (y + 0.5) * width.y / 5.0, (z + 0.5) * width.z / 5.0};
                    if (Contains(point, part.planes, 0.0)) probes.push_back(point);
                }
        const auto hits = Query(probes);
        const auto edges = PartEdges(part, tolerance_);
        if (edges.empty()) Numerical("convex cover part has no resolved edges");
        std::vector<std::vector<WorkPart>> alternatives(candidates.size());
        std::vector<WorkPart*> query_parts;
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& candidate = candidates[i];
            auto& children = alternatives[i];
            for (double sign : {1.0, -1.0}) {
                const Plane cut{candidate.normal * sign, candidate.offset * sign};
                Part child = CutPart(part, edges, cut, tolerance_);
                if (child.vertices.size() < 4u) { children.clear(); break; }
                children.push_back({std::move(child), {}});
            }
            for (auto& child : children) query_parts.push_back(&child);
        }
        PopulateInterior(query_parts);
        std::tuple<int, double, double> best{2, DBL_MAX, DBL_MAX};
        std::vector<WorkPart> selected;
        for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
            const auto& candidate = candidates[candidate_index];
            auto& alternative = alternatives[candidate_index];
            if (alternative.empty()) continue;
            std::vector<WorkPart> children;
            double maximum = 0.0, total = 0.0;
            for (size_t side = 0; side < alternative.size(); ++side) {
                const double sign = side == 0u ? 1.0 : -1.0;
                const Plane cut{candidate.normal * sign, candidate.offset * sign};
                auto& work = alternative[side];
                auto& child = work.geometry;
                auto& child_cloud = work.cloud;
                for (const auto& polygon : cloud.polygons) {
                    auto clipped = Clip(polygon, cut);
                    if (!clipped.vertices.empty()) child_cloud.polygons.push_back(std::move(clipped));
                }
                if (child_cloud.Empty()) continue;
                std::vector<std::pair<double, Plane>> additions;
                for (const auto& direction : directions) {
                    const double support = child_cloud.Support(direction).first + margin_;
                    double outer = -DBL_MAX;
                    for (const auto& vertex : child.vertices) outer = std::max(outer, Dot(direction, vertex));
                    bool existing = false;
                    for (auto& plane : child.planes)
                        if (Length(plane.normal - direction) < 1.0e-10) {
                            plane.offset = std::min(plane.offset, support);
                            existing = true;
                        }
                    if (!existing && outer - support > margin_ * 4.0)
                        additions.push_back({outer - support, {direction, support}});
                }
                std::stable_sort(additions.begin(), additions.end(),
                    [](const auto& l, const auto& r) { return l.first > r.first; });
                for (const auto& addition : additions)
                    if (child.planes.size() < params_.max_planes) AddPlane(child.planes, addition.second);
                double energy = 0.0;
                for (size_t i = 0; i < probes.size(); ++i)
                    if (Contains(probes[i], child.planes, margin_)) {
                        const double excess = std::max(0.0, double(hits[i].distance) - epsilon_);
                        energy += excess * excess;
                    }
                const auto child_bounds = Bounds(child.vertices);
                const Point radius = child_bounds.second - child_bounds.first;
                maximum = std::max(maximum, energy * Dot(radius, radius));
                total += energy;
                children.push_back(std::move(work));
            }
            if (children.empty()) continue;
            int unbroken = 0;
            if (!conflict.empty()) {
                double lo = DBL_MAX, hi = -DBL_MAX;
                for (const auto& point : conflict) {
                    const double value = Value(candidate, point);
                    lo = std::min(lo, value);
                    hi = std::max(hi, value);
                }
                unbroken = lo >= -margin_ || hi <= margin_ ? 1 : 0;
            }
            const auto score = std::make_tuple(unbroken, maximum, total);
            if (score < best) { best = score; selected = std::move(children); }
        }
        if (selected.empty()) Numerical("convex cover split could not retain a resolved solid");
        return selected;
    }

    collision::MeshSurfaceView source_;
    collision::MeshSurfaceInfo info_;
    const ConvexCoverParams& params_;
    MeshQueryBackend& queries_;
    ConvexCoverResult& result_;
    Point lower_, upper_;
    double margin_ = 0.0, tolerance_ = 0.0, epsilon_ = 0.0;
    std::vector<Polygon> source_polygons_;
    std::vector<Point> stencil_;
    std::map<CellKey, CellValue> cells_;
};

}  // namespace

ConvexCoverResult BuildConvexCover(collision::MeshSurfaceView source,
    collision::MeshSurfaceInfo info, const ConvexCoverParams& params, MeshQueryBackend& queries) {
    if (!collision::MeshSurfaceRangeValid(source, info) || !std::isfinite(params.relative_error) ||
        !(params.relative_error > 0.0) || params.max_parts == 0u || params.max_planes < 6u ||
        params.max_cells == 0u || params.max_operations == 0u)
        throw std::invalid_argument("Invalid convex cover input or budget");
    ConvexCoverResult result;
    result.backend = queries.Name();
    if (!(info.flags & collision::kMeshSurfaceClosed)) return result;
    try {
        Search search(source, info, params, queries, result);
        result.parts = search.Run();
        result.status = ConvexCoverStatus::Complete;
    } catch (const Unresolved& failure) {
        result.status = failure.status;
        result.reason = failure.what();
    } catch (const std::exception& failure) {
        result.status = ConvexCoverStatus::BackendFailure;
        result.reason = failure.what();
    }
    return result;
}

}  // namespace nuka::import::cooker
