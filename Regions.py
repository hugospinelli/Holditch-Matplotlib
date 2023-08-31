import numpy as np
import shapely
import shapely.ops
import shapely.validation


def _get_winding(parent_path, child):
    c = shapely.Polygon(child).representative_point()
    pc = np.array((c.x, c.y))
    ps = parent_path - pc
    ps = np.concatenate((ps, [ps[0]]))
    theta = np.sum(np.arctan2(
        np.cross(ps[:-1], ps[1:]),
        (ps[:-1]*ps[1:]).sum(-1)  # Inner product
    ))
    return round(theta/(2*np.pi))


def _get_shells(valid):
    if (isinstance(valid, shapely.MultiPolygon)
            or isinstance(valid, shapely.GeometryCollection)):
        shells = []
        for poly in valid.geoms:
            shells.extend(_get_shells(poly))
        return shells
        
    if not isinstance(valid, shapely.Polygon):
        return []
    
    if len(valid.interiors) == 0:
        return [valid]

    minx, miny, maxx, maxy = valid.bounds
    ps = [shapely.Point(minx, miny)]
    for ring in valid.interiors:
        ps.append(shapely.Polygon(ring).representative_point())
    ps.append(shapely.Point(maxx, maxy))
    gm = shapely.ops.split(valid, shapely.LineString(ps))
    return [poly for poly in gm.geoms if isinstance(poly, shapely.Polygon)]


def _to_list(geom):
    if isinstance(geom, shapely.GeometryCollection):
        return list(geom.geoms)
    if isinstance(geom, shapely.MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, shapely.Polygon):
        return [geom]
    return []


def _get_rect(outer_valid, inner_valid):
    minx1, miny1, maxx1, maxy1 = outer_valid.bounds
    minx2, miny2, maxx2, maxy2 = inner_valid.bounds
    minx = min(minx1, minx2)
    miny = min(miny1, miny2)
    maxx = max(maxx1, maxx2)
    maxy = max(maxy1, maxy2)
    ps = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    return shapely.Polygon(ps)


def _get_interiors(valid):
    interiors = []
    if isinstance(valid, shapely.Polygon):
        interiors = [shapely.Polygon(ring) for ring in valid.interiors]
    elif isinstance(valid, shapely.LinearRing):
        interiors = [shapely.Polygon(valid)]
    elif (isinstance(valid, shapely.MultiPolygon)
          or isinstance(valid, shapely.GeometryCollection)):
        for poly in valid.geoms:
            if isinstance(poly, shapely.Polygon):
                interiors.extend([shapely.Polygon(ring)
                                  for ring in poly.interiors])
    union = shapely.union_all(interiors)
    if isinstance(union, shapely.Polygon):
        union = [union]
    return shapely.MultiPolygon(union)


def _get_exteriors(valid):
    if isinstance(valid, shapely.Polygon):
        return shapely.MultiPolygon([shapely.Polygon(valid.exterior)])
    if (not (isinstance(valid, shapely.GeometryCollection)
             or isinstance(valid, shapely.MultiPolygon))):
        return shapely.MultiPolygon([])
    exteriors = []
    for poly in valid.geoms:
        if isinstance(poly, shapely.Polygon):
            exteriors.append(shapely.Polygon(poly.exterior))
        elif isinstance(poly, shapely.LinearRing):
            exteriors.append(shapely.Polygon(poly))
    union = shapely.union_all(exteriors)
    if isinstance(union, shapely.Polygon):
        union = [union]
    return shapely.MultiPolygon(union)


def _join_valids(outer_valid, inner_valid):
    holes1 = _get_interiors(outer_valid)
    holes2 = _get_interiors(inner_valid)
    holes = shapely.intersection(holes1, holes2)
    
    out = _get_rect(outer_valid, inner_valid)
    out = shapely.difference(out, outer_valid)
    out = shapely.difference(out, inner_valid)
    out = shapely.difference(out, holes)
    out = _to_list(out)
    holes = _to_list(holes)
    
    diff1 = _to_list(shapely.difference(outer_valid, inner_valid))
    diff2 = _to_list(shapely.difference(inner_valid, outer_valid))
    inter = _to_list(shapely.intersection(outer_valid, inner_valid))
    
    group = holes + out + diff1 + diff2 + inter
    return shapely.MultiPolygon(group)


def _paths_to_polygons(outer_path, inner_path):
    outer_poly = shapely.Polygon(outer_path)
    inner_poly = shapely.Polygon(inner_path)
    outer_valid = shapely.validation.make_valid(outer_poly)
    inner_valid = shapely.validation.make_valid(inner_poly)
    valids = _join_valids(outer_valid, inner_valid)
    shells = _get_shells(valids)
    return shells


def split_paths(outer_path, inner_path):
    polys = _paths_to_polygons(outer_path, inner_path)
    paths = []
    windings = []
    area = 0
    for child in polys:
        wo = _get_winding(outer_path, child)
        wi = _get_winding(inner_path, child)
        winding = wo - wi
        if winding == 0:
            continue
        paths.append(list(child.exterior.coords))
        windings.append(winding)
        area += winding*child.area
    return paths, windings, area

