// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#pragma once

#include "math_internal.h"
#include "solver.h"

typedef struct b3ManifoldConstraintPoint
{
	b3Vec3 rA, rB;
	// Angular velocity change per unit normal impulse. Preparation already
	// computes these products for effective mass; solver passes reuse them.
	b3Vec3 normalTorqueA, normalTorqueB;
	float baseSeparation;
	float relativeVelocity;
	float normalImpulse;
	float totalNormalImpulse;
	float normalMass;
	float leverArm;
	float cachedSeparation;
} b3ManifoldConstraintPoint;

typedef struct b3ManifoldConstraint
{
	b3ManifoldConstraintPoint points[4];
	int pointCount;
	b3Vec3 normal;
	b3Vec3 tangent1;
	b3Vec3 tangent2;
	b3Vec3 tangentTorqueA1, tangentTorqueA2;
	b3Vec3 tangentTorqueB1, tangentTorqueB2;
	b3Vec3 twistTorqueA, twistTorqueB;
	// Friction centers
	b3Vec3 centerA, centerB;
	float twistMass;
	float twistImpulse;
	b3Matrix2 tangentMass;
	b3Vec2 frictionImpulse;
	b3Vec3 rollingImpulse;
	float tangentVelocity1;
	float tangentVelocity2;
} b3ManifoldConstraint;

typedef struct b3ContactConstraint
{
	b3ManifoldConstraint* constraints;
	struct b3Contact* contact;
	int indexA;
	int indexB;
	float invMassA, invMassB;
	b3Matrix3 invIA, invIB;
	b3Softness softness;
	b3Matrix3 rollingMass;
	float friction;
	float restitution;
	float rollingResistance;
	int manifoldCount;
} b3ContactConstraint;

int b3GetWideContactConstraintByteCount( void );

// Overflow contacts don't fit into the constraint graph coloring
void b3PrepareContacts_Overflow( b3StepContext* context );
void b3WarmStartContacts_Overflow( b3StepContext* context );
void b3SolveContacts_Overflow( b3StepContext* context, bool useBias );
void b3ApplyRestitution_Overflow( b3StepContext* context );
void b3StoreImpulses_Overflow( b3StepContext* context );

void b3PrepareContacts_Mesh( b3SolverBlock block, b3StepContext* context );
void b3WarmStartContacts_Mesh( b3SolverBlock block, b3StepContext* context );
void b3SolveContacts_Mesh( b3SolverBlock block, b3StepContext* context, bool useBias );
void b3ApplyRestitution_Mesh( b3SolverBlock block, b3StepContext* context );
void b3StoreImpulses_Mesh( b3SolverBlock block, b3StepContext* context, int workerIndex );

void b3PrepareContacts_Convex( b3SolverBlock block, b3StepContext* context );
void b3WarmStartContacts_Convex( b3SolverBlock block, b3StepContext* context );
void b3SolveContacts_Convex( b3SolverBlock block, b3StepContext* context, bool useBias );
void b3ApplyRestitution_Convex( b3SolverBlock block, b3StepContext* context );
void b3StoreImpulses_Convex( b3SolverBlock block, b3StepContext* context, int workerIndex );
