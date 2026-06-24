#pragma once

#include <gtest/gtest.h>
#include <hiprt/hiprt.h>
#include <vector>

class CpuGeometryTest : public ::testing::Test
{
  protected:
	void SetUp() override;
	void TearDown() override;

	hiprtGeometry buildSampleTriangle(
		std::vector<hiprtFloat3>&	vertices,
		std::vector<uint32_t>&		indices,
		hiprtGeometryBuildInput&	geomInputOut );

	hiprtContext m_context = nullptr;
};
