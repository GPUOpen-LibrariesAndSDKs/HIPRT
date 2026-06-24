#pragma once

#include <gtest/gtest.h>
#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>
#include <vector>

class HybridGeometryTest : public ::testing::Test
{
  protected:
	void SetUp() override;
	void TearDown() override;

	hiprtGeometry buildSampleTriangle(
		std::vector<hiprtFloat3>&	vertices,
		std::vector<uint32_t>&		indices,
		hiprtGeometryBuildInput&	geomInputOut );

	hiprtContext m_context  = nullptr;
	oroCtx		 m_oroCtx   = nullptr;
	oroDevice	 m_oroDevice = 0;

	void freeManagedAllocs();
	std::vector<void*> m_managedAllocs;

	hiprtDevicePtr allocGeometryBuildTemp(
		const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions& buildOptions );
	hiprtDevicePtr allocSceneBuildTemp(
		const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions& buildOptions );
};
