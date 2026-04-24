/*
    Copyright (c) 2015-2017 Telecom ParisTech (France).
    Authors: Stephane Calderon and Tamy Boubekeur.
    All rights reserved.

    This file is part of Broxy, the reference implementation for
    the paper:
        Bounding Proxies for Shape Approximation.
        Stephane Calderon and Tamy Boubekeur.
        ACM Transactions on Graphics (Proc. SIGGRAPH 2017),
        vol. 36, no. 5, art. 57, 2017.

    You can redistribute it and/or modify it under the terms of the GNU
    General Public License as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.

    Licensees holding a valid commercial license may use this file in
    accordance with the commercial license agreement provided with the software.

    This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
    WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
*/

#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "Vec3.h"

namespace MorphoGraphics {
    /// Bounding Sphere simply defined by center and a radius
    class BoundingSphere {
    public:
        inline BoundingSphere () : _radius (0.f) {}
        inline BoundingSphere (const Vec3f & p) : _center (p), _radius (0.f) {}
        inline BoundingSphere (const Vec3f & center, float radius) : _center (center), _radius (radius) {}
        inline virtual ~BoundingSphere () {}
        inline void init (const Vec3f & p) { _center = p; }
        inline const Vec3f & center () const { return _center; }
        inline float radius () const { return _radius; }
        inline float volume () const { return float (4.0/3.0)*M_PI*_radius*_radius*_radius; }
        inline bool contains (const Vec3f & p) const { return ((p-_center).length () < _radius ); }
        inline bool contains (const BoundingSphere & b) const {
            if (!contains (b._center))
                return false;
            return  ((_radius - (b._center-_center).length ()) < b._radius);
        }
        inline void extendTo (const Vec3f & p) {
            if (contains (p))
                return;
            Vec3f x = _center-p;
            Vec3f d = x;
            d.normalize ();
            x += _radius*d;
            _radius = x.length ()/2.f;
            _center = p + x/2.f;
        }
        inline void extendTo (const BoundingSphere & b) {
            if (contains (b))
                return;
            Vec3f x = b._center - _center;
            Vec3f d = x;
            d.normalize ();
            Vec3f u = _center - _radius*d;
            Vec3f v = b._center + b._radius*d;
            _center = (u+v)/2.f;
            _radius = dist (u, v)/2.f;
        }
        inline void translate (const Vec3f & t) { _center += t; }
        inline void scale (float factor) { _radius *= factor; }
        inline bool intersect (const BoundingSphere & b) const { return ((_center-b._center).length () < (_radius + b._radius)); }
    private:
        Vec3f _center;
        float _radius;
    };
        
    /// Axis-aligned bounding box class with operators
    class AxisAlignedBoundingBox {
    public:
        inline AxisAlignedBoundingBox () {}
        inline AxisAlignedBoundingBox (const Vec3f & p) : _min (p), _max (p) {}
        inline AxisAlignedBoundingBox (const Vec3f & minP, const Vec3f & maxP) : _min (minP), _max (maxP) {}
        inline virtual ~AxisAlignedBoundingBox () {}
        inline void init (const Vec3f & p) { _min = _max = p; }
        inline const Vec3f & min () const { return _min; }
        inline const Vec3f & max () const { return _max; }
        inline Vec3f center () const { return (_min + _max) / 2; }
        inline float width () const { return (_max[0] - _min[0]); }
        inline float height () const { return (_max[1] - _min[1]); }
        inline float length () const { return (_max[2] - _min[2]); }
        inline float size () const { return std::max (width (), std::max (height (), length ())); }
        inline float radius () const { return dist (_min, _max) / 2.0; }
        inline float volume () const { return width () * height () * length (); }
        inline bool contains (const Vec3f & p) const {
            for (unsigned int i = 0; i < 3; i++)
                if (!(p[i] >= _min[i] && p[i] <= _max[i]))
                    return false;
            return true;
        }
        inline bool contains (const AxisAlignedBoundingBox & b) const { return contains (b._min) && contains (b._max); }
        inline void extendTo (const Vec3f & p) {
            for (unsigned int i = 0; i < 3; i++) {
                if (p[i] > _max[i])
                    _max[i] = p[i];
                if (p[i] < _min[i])
                    _min[i] = p[i];
            }
        }
        inline void extendTo (const AxisAlignedBoundingBox & b) {
            extendTo (b._min);
            extendTo (b._max);
        }
        inline void extend (float offset) {
            Vec3f delta (offset, offset, offset);
            _min -= delta;
            _max += delta;
        }
        inline void translate (const Vec3f & t) {
            _min += t;
            _max += t;
        }
        inline void scale (float factor) {
            Vec3f c = center ();
            _min = c + factor * (_min - c);
            _max = c + factor * (_max - c);
        }
        inline bool intersect (const AxisAlignedBoundingBox & b) const {
            for (unsigned int i = 0; i < 3; i++)
                if (_max[i] < b._min[i] || _min[i] > b._max[i])
                    return false; 
            return true; 
        }
    private:
        inline float whl (unsigned int i) const { return (_max[i] - _min[i]); }
        inline float middle (unsigned int i) const { return ((_min[i] + _max[i]) / 2.0); }
        static inline bool isIn (float x, float min, float max) { return (x >= min && x <= max); }
        Vec3f _min, _max;
    };
};


// Some Emacs-Hints -- please don't remove:
//
//  Local Variables:
//  mode:C++
//  tab-width:4
//  End:
