// *****************************************************************************
//
// Copyright (c) 2026, Southwest Research Institute® (SwRI®)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Southwest Research Institute® (SwRI®) nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// *****************************************************************************

#include <gtest/gtest.h>
#include <QApplication>
#include <QCoreApplication>
#include <QLineEdit>
#include <QFont>
#include <QOpenGLWidget>
#include <QString>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#include <marti_common_msgs/msg/float32_stamped.hpp>
#include <marti_common_msgs/msg/string_stamped.hpp>
#include <marti_visualization_msgs/msg/textured_marker.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <mapviz_plugins/attitude_indicator_plugin.hpp>
#include <mapviz_plugins/disparity_plugin.hpp>
#include <mapviz_plugins/draw_marker_plugin.hpp>
#include <mapviz_plugins/draw_polygon_plugin.hpp>
#include <mapviz_plugins/float_plugin.hpp>
#include <mapviz_plugins/gps_plugin.hpp>
#include <mapviz_plugins/image_plugin.hpp>
#include <mapviz_plugins/laserscan_plugin.hpp>
#include <mapviz_plugins/marker_plugin.hpp>
#include <mapviz_plugins/measuring_plugin.hpp>
#include <mapviz_plugins/navsat_plugin.hpp>
#include <mapviz_plugins/occupancy_grid_plugin.hpp>
#include <mapviz_plugins/odometry_plugin.hpp>
#include <mapviz_plugins/path_plugin.hpp>
#include <mapviz_plugins/point_click_publisher_plugin.hpp>
#include <mapviz_plugins/pointcloud2_plugin.hpp>
#include <mapviz_plugins/pose_plugin.hpp>
#include <mapviz_plugins/robot_model_plugin.hpp>
#include <mapviz_plugins/route_plugin.hpp>
#include <mapviz_plugins/speedometer_plugin.hpp>
#include <mapviz_plugins/string_plugin.hpp>
#include <mapviz_plugins/textured_marker_plugin.hpp>

namespace
{
/// Shared by every test.  Plugins subscribe while loading their config, so
/// they need a node just as they get one from Mapviz::CreateNewDisplay(),
/// which calls SetNode() before LoadConfigPlugin().
rclcpp::Node::SharedPtr g_node;

/// Stands in for the map canvas.  It is never shown, so no OpenGL context is
/// needed; plugins only ask it to repaint.
std::unique_ptr<QOpenGLWidget> g_canvas;

/// PointCloud2Plugin repaints while loading its config, which needs the
/// canvas that Initialize() normally provides.  Initialize() also needs an
/// OpenGL context, so hand the canvas over directly.
class CanvasPointCloud2Plugin : public mapviz_plugins::PointCloud2Plugin
{
public:
  CanvasPointCloud2Plugin() {canvas_ = g_canvas.get();}
};

/// Round trips a plugin's configuration the way mapviz does when a config
/// file is opened and then saved again.
YAML::Node SaveAfterLoading(mapviz::MapvizPlugin & plugin, const std::string & yaml)
{
  plugin.LoadConfigPlugin(YAML::Load(yaml), "");

  YAML::Emitter emitter;
  emitter << YAML::BeginMap;
  plugin.SaveConfigPlugin(emitter, "");
  emitter << YAML::EndMap;

  return YAML::Load(emitter.c_str());
}

/// Loads a config that sets a single topic key, the same path mapviz takes
/// when a config file is opened.
void LoadTopic(mapviz::MapvizPlugin & plugin, const std::string & key, const std::string & topic)
{
  YAML::Node node;
  node[key] = topic;
  plugin.LoadConfigPlugin(node, "");
}

/// Subscriptions show up in the ROS graph asynchronously on some middleware,
/// and some plugins subscribe from a Qt timer, so poll briefly before deciding.
::testing::AssertionResult HasSubscribers(const std::string & topic)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (g_node->count_subscribers(topic) == 0) {
    if (std::chrono::steady_clock::now() > deadline) {
      return ::testing::AssertionFailure() << "no subscribers on " << topic;
    }
    QCoreApplication::processEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult HasNoSubscribers(const std::string & topic)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  size_t count = 0;
  while ((count = g_node->count_subscribers(topic)) != 0) {
    if (std::chrono::steady_clock::now() > deadline) {
      return ::testing::AssertionFailure() << count << " subscriber(s) left on " << topic;
    }
    QCoreApplication::processEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return ::testing::AssertionSuccess();
}

struct TopicCase
{
  std::string name;
  std::string key;
  std::function<std::unique_ptr<mapviz::MapvizPlugin>()> create;
  /// Set for plugins that accept several message types and pick one from the
  /// topic's publishers, so they only subscribe once something is publishing.
  std::function<rclcpp::PublisherBase::SharedPtr(const std::string &)> advertise;
};

template<typename PluginT>
TopicCase Case(const std::string & name, const std::string & key = "topic")
{
  return {name, key, [] {return std::make_unique<PluginT>();}, nullptr};
}

template<typename PluginT, typename MsgT>
TopicCase AdvertisedCase(const std::string & name)
{
  return {
    name, "topic",
    [] {return std::make_unique<PluginT>();},
    [](const std::string & topic) {return g_node->create_publisher<MsgT>(topic, 1);}};
}

void PrintTo(const TopicCase & topic_case, std::ostream * os)
{
  *os << topic_case.name;
}
}  // namespace

/// Loads @p font into a new @p PluginT and reads back the font it saves.
template<typename PluginT>
QFont FontAfterLoading(const std::string & font)
{
  PluginT plugin;
  plugin.SetNode(*g_node);
  YAML::Node saved = SaveAfterLoading(plugin, "font: '" + font + "'\n");
  QFont loaded;
  loaded.fromString(QString::fromStdString(saved["font"].as<std::string>()));
  return loaded;
}

TEST(FontConfig, LoadsFontsSavedByEitherQtVersion)
{
  // Qt 5 couldn't read fonts saved by Qt 6, so configs moved from a Qt 6
  // distro to a Qt 5 one lost their fonts.
  for (const std::string font : {
    "Monospace,26,-1,5,50,0,0,0,0,0",
    "Monospace,26,-1,5,400,0,0,0,0,0,0,0,0,0,0,1"})
  {
    const QFont float_font = FontAfterLoading<mapviz_plugins::FloatPlugin>(font);
    EXPECT_EQ("Monospace", float_font.family()) << "float: " << font;
    EXPECT_EQ(26, float_font.pointSize()) << "float: " << font;

    // The string plugin keeps its own font size.
    const QFont string_font = FontAfterLoading<mapviz_plugins::StringPlugin>(font);
    EXPECT_EQ("Monospace", string_font.family()) << "string: " << font;
  }
}

TEST(SpeedometerConfig, RoundTripsSettings)
{
  mapviz_plugins::SpeedometerPlugin plugin;
  plugin.SetNode(*g_node);

  YAML::Node saved = SaveAfterLoading(
    plugin,
    "topic: /vehicle/odom\n"
    "max_speed: 33.5\n"
    "color: '#ff00ff'\n"
    "x: 40\n"
    "y: 55\n"
    "width: 180\n"
    "height: 190\n");

  EXPECT_EQ("/vehicle/odom", saved["topic"].as<std::string>());
  EXPECT_DOUBLE_EQ(33.5, saved["max_speed"].as<double>());
  EXPECT_EQ("#ff00ff", saved["color"].as<std::string>());
  EXPECT_EQ(40, saved["x"].as<int>());
  EXPECT_EQ(55, saved["y"].as<int>());
  EXPECT_EQ(180, saved["width"].as<int>());
  EXPECT_EQ(190, saved["height"].as<int>());
}

TEST(SpeedometerConfig, UsesDefaultMaxSpeedWhenUnset)
{
  mapviz_plugins::SpeedometerPlugin plugin;
  plugin.SetNode(*g_node);

  YAML::Node saved = SaveAfterLoading(plugin, "topic: /odom\n");

  EXPECT_DOUBLE_EQ(40.0, saved["max_speed"].as<double>());
}

TEST(SpeedometerConfig, ClampsOutOfRangeMaxSpeed)
{
  mapviz_plugins::SpeedometerPlugin plugin;
  plugin.SetNode(*g_node);

  // A hand edited config must not be able to push the dial outside the range
  // the config widget allows.
  EXPECT_DOUBLE_EQ(
    1000.0,
    SaveAfterLoading(plugin, "max_speed: 99999999\n")["max_speed"].as<double>());
  EXPECT_DOUBLE_EQ(
    0.1,
    SaveAfterLoading(plugin, "max_speed: -5\n")["max_speed"].as<double>());
}

TEST(DrawMarkerConfig, RoundTripsSettings)
{
  mapviz_plugins::DrawMarkerPlugin plugin;
  plugin.SetNode(*g_node);

  YAML::Node saved = SaveAfterLoading(
    plugin,
    "frame: map\n"
    "topic: /drawn\n"
    "type: 2\n"
    "namespace: shapes\n"
    "id: 7\n"
    "color: '#00ffff'\n"
    "alpha: 0.5\n"
    "scale: 2.5\n");

  EXPECT_EQ("map", saved["frame"].as<std::string>());
  EXPECT_EQ("/drawn", saved["topic"].as<std::string>());
  EXPECT_EQ(2, saved["type"].as<int>());
  EXPECT_EQ("shapes", saved["namespace"].as<std::string>());
  EXPECT_EQ(7, saved["id"].as<int>());
  EXPECT_EQ("#00ffff", saved["color"].as<std::string>());
  EXPECT_DOUBLE_EQ(0.5, saved["alpha"].as<double>());
  EXPECT_DOUBLE_EQ(2.5, saved["scale"].as<double>());
}

TEST(DrawMarkerConfig, RoundTripsVertices)
{
  mapviz_plugins::DrawMarkerPlugin plugin;
  plugin.SetNode(*g_node);

  // Persisting the vertices is what lets a drawing outlive the session it was
  // made in, so the coordinates have to survive exactly.
  YAML::Node saved = SaveAfterLoading(
    plugin,
    "vertices:\n"
    "  - [1.5, -2.5]\n"
    "  - [3.0, 4.0]\n"
    "  - [-5.25, 6.125]\n");

  ASSERT_EQ(3u, saved["vertices"].size());
  EXPECT_DOUBLE_EQ(1.5, saved["vertices"][0][0].as<double>());
  EXPECT_DOUBLE_EQ(-2.5, saved["vertices"][0][1].as<double>());
  EXPECT_DOUBLE_EQ(3.0, saved["vertices"][1][0].as<double>());
  EXPECT_DOUBLE_EQ(4.0, saved["vertices"][1][1].as<double>());
  EXPECT_DOUBLE_EQ(-5.25, saved["vertices"][2][0].as<double>());
  EXPECT_DOUBLE_EQ(6.125, saved["vertices"][2][1].as<double>());
}

TEST(DrawMarkerConfig, HandlesConfigWithoutVertices)
{
  mapviz_plugins::DrawMarkerPlugin plugin;
  plugin.SetNode(*g_node);

  // Configs written before vertices were persisted have no such key.
  YAML::Node saved = SaveAfterLoading(plugin, "frame: map\n");

  ASSERT_TRUE(saved["vertices"]);
  EXPECT_EQ(0u, saved["vertices"].size());
}

TEST(DrawMarkerConfig, IgnoresMalformedVertices)
{
  mapviz_plugins::DrawMarkerPlugin plugin;
  plugin.SetNode(*g_node);

  // A vertex needs both coordinates; a short one is skipped rather than read
  // out of bounds.
  YAML::Node saved = SaveAfterLoading(
    plugin,
    "vertices:\n"
    "  - [1.0, 2.0]\n"
    "  - [3.0]\n"
    "  - [4.0, 5.0]\n");

  ASSERT_EQ(2u, saved["vertices"].size());
  EXPECT_DOUBLE_EQ(1.0, saved["vertices"][0][0].as<double>());
  EXPECT_DOUBLE_EQ(4.0, saved["vertices"][1][0].as<double>());
}

class TopicEditing : public ::testing::TestWithParam<TopicCase> {};

TEST_P(TopicEditing, FollowsTheConfiguredTopic)
{
  const TopicCase & topic_case = GetParam();
  const std::string first = "/test_topic_editing/" + topic_case.name + "/first";
  const std::string second = "/test_topic_editing/" + topic_case.name + "/second";

  rclcpp::PublisherBase::SharedPtr first_pub, second_pub;
  if (topic_case.advertise) {
    first_pub = topic_case.advertise(first);
    second_pub = topic_case.advertise(second);
  }

  std::unique_ptr<mapviz::MapvizPlugin> plugin = topic_case.create();
  plugin->SetNode(*g_node);

  LoadTopic(*plugin, topic_case.key, first);
  EXPECT_TRUE(HasSubscribers(first));

  // Changing the topic has to move the subscription, not add a second one or
  // leave the old one in place (#914, #918).
  LoadTopic(*plugin, topic_case.key, second);
  EXPECT_TRUE(HasSubscribers(second));
  EXPECT_TRUE(HasNoSubscribers(first));

  // Clearing the field unsubscribes.  rclcpp rejects an empty topic name, so
  // this must not reach create_subscription() (#913).
  EXPECT_NO_THROW(LoadTopic(*plugin, topic_case.key, ""));
  EXPECT_TRUE(HasNoSubscribers(second));

  // Entering the previous topic again has to subscribe again, even though it
  // matches what the plugin last subscribed to.
  LoadTopic(*plugin, topic_case.key, second);
  EXPECT_TRUE(HasSubscribers(second));

  // A name ROS rejects is reported instead of thrown, and drops the previous
  // subscription like any other change.
  EXPECT_NO_THROW(LoadTopic(*plugin, topic_case.key, "/test_topic_editing/not a topic"));
  EXPECT_TRUE(HasNoSubscribers(second));

  // Correcting it subscribes again.
  LoadTopic(*plugin, topic_case.key, second);
  EXPECT_TRUE(HasSubscribers(second));
}

INSTANTIATE_TEST_SUITE_P(
  Plugins, TopicEditing,
  ::testing::Values(
    AdvertisedCase<mapviz_plugins::AttitudeIndicatorPlugin, sensor_msgs::msg::Imu>(
      "attitude_indicator"),
    Case<mapviz_plugins::DisparityPlugin>("disparity"),
    AdvertisedCase<mapviz_plugins::FloatPlugin, std_msgs::msg::Float64>("float"),
    Case<mapviz_plugins::GpsPlugin>("gps"),
    Case<mapviz_plugins::ImagePlugin>("image"),
    Case<mapviz_plugins::LaserScanPlugin>("laserscan"),
    AdvertisedCase<mapviz_plugins::MarkerPlugin, visualization_msgs::msg::Marker>("marker"),
    Case<mapviz_plugins::NavSatPlugin>("navsat"),
    Case<mapviz_plugins::OccupancyGridPlugin>("occupancy_grid"),
    Case<mapviz_plugins::OdometryPlugin>("odometry"),
    Case<mapviz_plugins::PathPlugin>("path"),
    Case<CanvasPointCloud2Plugin>("pointcloud2"),
    Case<mapviz_plugins::PosePlugin>("pose"),
    Case<mapviz_plugins::RobotModelPlugin>("robot_model"),
    Case<mapviz_plugins::RoutePlugin>("route"),
    Case<mapviz_plugins::RoutePlugin>("route_position", "postopic"),
    Case<mapviz_plugins::SpeedometerPlugin>("speedometer"),
    AdvertisedCase<mapviz_plugins::StringPlugin, std_msgs::msg::String>("string"),
    AdvertisedCase<
      mapviz_plugins::TexturedMarkerPlugin,
      marti_visualization_msgs::msg::TexturedMarker>("textured_marker")),
  [](const ::testing::TestParamInfo<TopicCase> & info) {return info.param.name;});

TEST(MultiTypeTopics, SubscribesOnlyWithThePublishedType)
{
  const std::string topic = "/test_multi_type_topics/stamped_float";
  auto pub = g_node->create_publisher<marti_common_msgs::msg::Float32Stamped>(topic, 1);

  mapviz_plugins::FloatPlugin plugin;
  plugin.SetNode(*g_node);

  // Subscribing to the topic under every supported type makes Fast DDS throw
  // "incompatible type", so exactly one subscription, of the published type,
  // is expected.
  EXPECT_NO_THROW(LoadTopic(plugin, "topic", topic));
  ASSERT_TRUE(HasSubscribers(topic));
  EXPECT_EQ(1u, g_node->count_subscribers(topic));
}

TEST(MultiTypeTopics, SubscribesToStampedStrings)
{
  const std::string topic = "/test_multi_type_topics/stamped_string";
  auto pub = g_node->create_publisher<marti_common_msgs::msg::StringStamped>(topic, 1);

  mapviz_plugins::StringPlugin plugin;
  plugin.SetNode(*g_node);

  LoadTopic(plugin, "topic", topic);
  ASSERT_TRUE(HasSubscribers(topic));
  EXPECT_EQ(1u, g_node->count_subscribers(topic));
}

TEST(MultiTypeTopics, SubscribesOnceAPublisherAppears)
{
  const std::string topic = "/test_multi_type_topics/late_publisher";

  mapviz_plugins::FloatPlugin plugin;
  plugin.SetNode(*g_node);

  // Mapviz is often started before the rest of the system.
  LoadTopic(plugin, "topic", topic);
  EXPECT_TRUE(HasNoSubscribers(topic));

  auto pub = g_node->create_publisher<std_msgs::msg::Float64>(topic, 1);
  EXPECT_TRUE(HasSubscribers(topic));
}

/// Records what a plugin reports through PrintError(), and exposes the
/// protected publishing slots of the plugins that have them.
template<typename PluginT>
class ErrorCapture : public PluginT
{
public:
  void PrintError(const std::string & message) override
  {
    errors.push_back(message);
    PluginT::PrintError(message);
  }

  std::vector<std::string> errors;
};

class PublishingDrawMarker : public ErrorCapture<mapviz_plugins::DrawMarkerPlugin>
{
public:
  using DrawMarkerPlugin::PublishMarker;
};

class PublishingDrawPolygon : public ErrorCapture<mapviz_plugins::DrawPolygonPlugin>
{
public:
  using DrawPolygonPlugin::PublishPolygon;
};

::testing::AssertionResult ReportedInvalid(
  const std::vector<std::string> & errors, const std::string & kind)
{
  for (const std::string & error : errors) {
    if (error.rfind("Invalid " + kind, 0) == 0) {
      return ::testing::AssertionSuccess();
    }
  }
  ::testing::AssertionResult result = ::testing::AssertionFailure();
  result << "no \"Invalid " << kind << "\" error among " << errors.size() << ":";
  for (const std::string & error : errors) {
    result << " [" << error << "]";
  }
  return result;
}

TEST(InvalidNames, ReportsARejectedSubscriptionTopic)
{
  ErrorCapture<mapviz_plugins::PosePlugin> plugin;
  plugin.SetNode(*g_node);

  EXPECT_NO_THROW(LoadTopic(plugin, "topic", "/test_invalid_names/not a topic"));

  EXPECT_TRUE(ReportedInvalid(plugin.errors, "topic name"));
}

TEST(InvalidNames, ReportsARejectedPublisherTopic)
{
  ErrorCapture<mapviz_plugins::PointClickPublisherPlugin> plugin;
  plugin.SetNode(*g_node);

  EXPECT_NO_THROW(LoadTopic(plugin, "topic", "/test_invalid_names/not a topic"));

  EXPECT_TRUE(ReportedInvalid(plugin.errors, "topic name"));
}

TEST(InvalidNames, DrawMarkerDoesNotPublishToARejectedTopic)
{
  PublishingDrawMarker plugin;
  plugin.SetNode(*g_node);

  // Empty topics are rejected too, and used to throw here as well.
  for (const std::string topic : {"/test_invalid_names/not a topic", ""}) {
    SCOPED_TRACE("topic '" + topic + "'");
    plugin.errors.clear();
    LoadTopic(plugin, "topic", topic);
    EXPECT_NO_THROW(plugin.PublishMarker());
    EXPECT_TRUE(ReportedInvalid(plugin.errors, "topic name"));
  }
}

TEST(InvalidNames, DrawPolygonDoesNotPublishToARejectedTopic)
{
  PublishingDrawPolygon plugin;
  plugin.SetNode(*g_node);

  for (const std::string topic : {"/test_invalid_names/not a topic", ""}) {
    SCOPED_TRACE("topic '" + topic + "'");
    plugin.errors.clear();
    LoadTopic(plugin, "polygon_topic", topic);
    EXPECT_NO_THROW(plugin.PublishPolygon());
    EXPECT_TRUE(ReportedInvalid(plugin.errors, "topic name"));
  }
}

TEST(RouteTopics, SubscribesToThePositionTopicWithoutARouteTopic)
{
  mapviz_plugins::RoutePlugin plugin;
  plugin.SetNode(*g_node);

  // The position subscription used to be created on the route topic, which
  // is empty here, so the whole load failed with "topic name must not be
  // empty string" (#913).
  EXPECT_NO_THROW(
    plugin.LoadConfigPlugin(
      YAML::Load(
        "topic: ''\n"
        "postopic: /test_route_topics/position\n"),
      ""));

  EXPECT_TRUE(HasSubscribers("/test_route_topics/position"));
}

TEST(RouteTopics, KeepsRouteAndPositionSubscriptionsApart)
{
  mapviz_plugins::RoutePlugin plugin;
  plugin.SetNode(*g_node);

  plugin.LoadConfigPlugin(
    YAML::Load(
      "topic: /test_route_topics/apart/route\n"
      "postopic: /test_route_topics/apart/position\n"),
    "");

  // One subscription each; a mix-up puts both on the same topic.
  EXPECT_EQ(1u, g_node->count_subscribers("/test_route_topics/apart/route"));
  EXPECT_EQ(1u, g_node->count_subscribers("/test_route_topics/apart/position"));
}

TEST(OccupancyGridTopics, SubscribesToUpdatesOnlyWhenChecked)
{
  const std::string grid = "/test_occupancy_grid_topics/map";
  const std::string updates = grid + "_updates";

  mapviz_plugins::OccupancyGridPlugin plugin;
  plugin.SetNode(*g_node);

  // Updates arrive on "<topic>_updates", never on the grid topic itself
  // (#918).
  plugin.LoadConfigPlugin(YAML::Load("topic: " + grid + "\nupdate: true\n"), "");
  EXPECT_TRUE(HasSubscribers(grid));
  EXPECT_TRUE(HasSubscribers(updates));
  EXPECT_EQ(1u, g_node->count_subscribers(grid));

  // Unchecking used to resubscribe instead of unsubscribing (#918).
  plugin.LoadConfigPlugin(YAML::Load("update: false\n"), "");
  EXPECT_TRUE(HasNoSubscribers(updates));
  EXPECT_TRUE(HasSubscribers(grid));

  plugin.LoadConfigPlugin(YAML::Load("update: true\n"), "");
  EXPECT_TRUE(HasSubscribers(updates));
}

TEST(OccupancyGridTopics, DoesNotSubscribeToUpdatesWithoutAGridTopic)
{
  mapviz_plugins::OccupancyGridPlugin plugin;
  plugin.SetNode(*g_node);

  // Checking the box with no topic must not subscribe to a bare "_updates".
  EXPECT_NO_THROW(plugin.LoadConfigPlugin(YAML::Load("topic: ''\nupdate: true\n"), ""));
  EXPECT_TRUE(HasNoSubscribers("/_updates"));
}

namespace mapviz_plugins
{
class MeasuringPluginTest : public ::testing::Test
{
protected:
  /// Places vertices as if they had been clicked, then measures.
  static void Measure(MeasuringPlugin & plugin, const std::vector<tf2::Vector3> & vertices)
  {
    plugin.vertices_ = vertices;
    plugin.DistanceCalculation();
  }

  static const std::vector<double> & Measurements(const MeasuringPlugin & plugin)
  {
    return plugin.measurements_;
  }

  static std::string TotalText(const MeasuringPlugin & plugin)
  {
    return plugin.ui_.totaldistance->text().toStdString();
  }

  static std::string LastSegmentText(const MeasuringPlugin & plugin)
  {
    return plugin.ui_.measurement->text().toStdString();
  }
};
}  // namespace mapviz_plugins

using mapviz_plugins::MeasuringPluginTest;

TEST_F(MeasuringPluginTest, CountsAPointAtTheOrigin)
{
  mapviz_plugins::MeasuringPlugin plugin;
  plugin.SetNode(*g_node);

  // The origin used to double as "no previous point", so a click at (0, 0)
  // was left out of the distance (#924).
  Measure(plugin, {tf2::Vector3(0, 0, 0), tf2::Vector3(3, 4, 0)});

  ASSERT_EQ(2u, Measurements(plugin).size());
  EXPECT_DOUBLE_EQ(5.0, Measurements(plugin)[0]);
  EXPECT_DOUBLE_EQ(5.0, Measurements(plugin).back());
  EXPECT_EQ("5 meters", TotalText(plugin));
  EXPECT_EQ("5 meters", LastSegmentText(plugin));
}

TEST_F(MeasuringPluginTest, CountsTheOriginInTheMiddleOfAPath)
{
  mapviz_plugins::MeasuringPlugin plugin;
  plugin.SetNode(*g_node);

  Measure(
    plugin,
    {tf2::Vector3(3, 4, 0), tf2::Vector3(0, 0, 0), tf2::Vector3(0, 2, 0)});

  // One measurement per segment, then the total.
  ASSERT_EQ(3u, Measurements(plugin).size());
  EXPECT_DOUBLE_EQ(5.0, Measurements(plugin)[0]);
  EXPECT_DOUBLE_EQ(2.0, Measurements(plugin)[1]);
  EXPECT_DOUBLE_EQ(7.0, Measurements(plugin)[2]);
  EXPECT_EQ("7 meters", TotalText(plugin));
  EXPECT_EQ("2 meters", LastSegmentText(plugin));
}

TEST_F(MeasuringPluginTest, ShowsNothingForASinglePoint)
{
  mapviz_plugins::MeasuringPlugin plugin;
  plugin.SetNode(*g_node);

  Measure(plugin, {tf2::Vector3(1, 1, 0)});

  ASSERT_EQ(1u, Measurements(plugin).size());
  EXPECT_DOUBLE_EQ(0.0, Measurements(plugin)[0]);
  EXPECT_EQ("", TotalText(plugin));
  EXPECT_EQ("", LastSegmentText(plugin));
}

/// Records the marker plugin's status line.
class MarkerStatus : public mapviz_plugins::MarkerPlugin
{
public:
  void PrintError(const std::string & message) override
  {
    status = "error: " + message;
    ++errors;
    MarkerPlugin::PrintError(message);
  }
  void PrintWarning(const std::string & message) override
  {
    status = "warning: " + message;
    MarkerPlugin::PrintWarning(message);
  }
  void PrintInfo(const std::string & message) override
  {
    status = "info: " + message;
    MarkerPlugin::PrintInfo(message);
  }

  std::string status;
  int errors = 0;
};

/// Runs the Qt event loop, and so the plugins' 1 s timers, until @p done or @p timeout.
::testing::AssertionResult ProcessEventsUntil(
  const std::function<bool()> & done, std::chrono::milliseconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() > deadline) {
      return ::testing::AssertionFailure() << "timed out";
    }
    try {
      QCoreApplication::processEvents();
    } catch (const std::exception & e) {
      return ::testing::AssertionFailure() << "an event handler threw: " << e.what();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return ::testing::AssertionSuccess();
}

/// Runs the Qt event loop for @p duration; fails if an event handler throws.
::testing::AssertionResult ProcessEventsFor(std::chrono::milliseconds duration)
{
  const auto end = std::chrono::steady_clock::now() + duration;
  while (std::chrono::steady_clock::now() < end) {
    try {
      QCoreApplication::processEvents();
    } catch (const std::exception & e) {
      return ::testing::AssertionFailure() << "an event handler threw: " << e.what();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return ::testing::AssertionSuccess();
}

/// Loads a marker display saved with @p saved_type while its topic is published
/// as PublishedT, and waits for the display to switch to the published type.
template<typename PublishedT>
void ExpectMarkerTypeCorrected(
  const std::string & topic, const std::string & saved_type, const std::string & published_type)
{
  // Fast DDS refuses two types on one topic within a participant, so the
  // publisher gets a context, and so a participant, of its own.
  auto context = std::make_shared<rclcpp::Context>();
  context->init(0, nullptr);
  auto publisher_node = std::make_shared<rclcpp::Node>(
    "test_marker_type_publisher", rclcpp::NodeOptions().context(context));

  mapviz_plugins::MarkerPlugin plugin;
  plugin.SetNode(*g_node);

  // Mapviz started before the publisher: the display subscribes with the saved
  // type, which puts that type in the topic list next to the published one.
  YAML::Node config;
  config["topic"] = topic;
  config["message_type"] = saved_type;
  plugin.LoadConfigPlugin(config, "");
  ASSERT_TRUE(HasSubscribers(topic));

  auto publisher = publisher_node->create_publisher<PublishedT>(topic, 10);

  // The display checks from a 1 s Qt timer.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (publisher->get_subscription_count() == 0 &&
    std::chrono::steady_clock::now() < deadline)
  {
    QCoreApplication::processEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_EQ(1u, publisher->get_subscription_count());

  YAML::Emitter emitter;
  emitter << YAML::BeginMap;
  plugin.SaveConfigPlugin(emitter, "");
  emitter << YAML::EndMap;
  EXPECT_EQ(published_type, YAML::Load(emitter.c_str())["message_type"].as<std::string>(""));

  publisher.reset();
  publisher_node.reset();
  context->shutdown("test finished");
}

TEST(MarkerType, CorrectsASavedMarkerTypeToMarkerArray)
{
  ExpectMarkerTypeCorrected<visualization_msgs::msg::MarkerArray>(
    "/test_marker_type/saved_marker",
    "visualization_msgs/msg/Marker", "visualization_msgs/msg/MarkerArray");
}

TEST(MarkerType, CorrectsASavedMarkerArrayTypeToMarker)
{
  ExpectMarkerTypeCorrected<visualization_msgs::msg::Marker>(
    "/test_marker_type/saved_marker_array",
    "visualization_msgs/msg/MarkerArray", "visualization_msgs/msg/Marker");
}

TEST(MarkerType, SwitchesTwoDisplaysOnOneTopic)
{
  // Two displays saved as Marker on a topic now published as MarkerArray. The first to
  // switch waits while the other still subscribes as Marker, then follows it.
  const std::string topic = "/test_marker_type/two_displays";
  auto context = std::make_shared<rclcpp::Context>();
  context->init(0, nullptr);
  auto publisher_node = std::make_shared<rclcpp::Node>(
    "test_marker_type_publisher", rclcpp::NodeOptions().context(context));

  YAML::Node config;
  config["topic"] = topic;
  config["message_type"] = "visualization_msgs/msg/Marker";
  MarkerStatus first;
  first.SetNode(*g_node);
  first.LoadConfigPlugin(config, "");
  MarkerStatus second;
  second.SetNode(*g_node);
  second.LoadConfigPlugin(config, "");
  ASSERT_TRUE(HasSubscribers(topic));

  auto publisher =
    publisher_node->create_publisher<visualization_msgs::msg::MarkerArray>(topic, 10);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (publisher->get_subscription_count() < 2 &&
    std::chrono::steady_clock::now() < deadline)
  {
    ASSERT_NO_THROW(QCoreApplication::processEvents());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_EQ(2u, publisher->get_subscription_count());
  // The wait's warning is replaced once subscribed.
  EXPECT_EQ("warning: No messages received.", first.status);
  EXPECT_EQ("warning: No messages received.", second.status);

  publisher.reset();
  publisher_node.reset();
  context->shutdown("test finished");
}

TEST(MarkerType, ForgetsTheTypeWhenAReloadChangesTheTopic)
{
  // AddMapvizDisplay on an existing display reloads its config. A new topic without
  // message_type must not inherit the old topic's type.
  const std::string topic = "/test_marker_type/reloaded";
  mapviz_plugins::MarkerPlugin plugin;
  plugin.SetNode(*g_node);
  SaveAfterLoading(
    plugin,
    "topic: /test_marker_type/before_reload\n"
    "message_type: visualization_msgs/msg/MarkerArray\n");

  YAML::Node saved = SaveAfterLoading(plugin, "topic: " + topic + "\n");

  EXPECT_FALSE(saved["message_type"]);
  EXPECT_TRUE(HasNoSubscribers(topic));  // waits for the topic to be advertised
}

TEST(MarkerType, KeepsTheTypeWhenAReloadKeepsTheTopic)
{
  const std::string topic = "/test_marker_type/reloaded_same";
  mapviz_plugins::MarkerPlugin plugin;
  plugin.SetNode(*g_node);
  SaveAfterLoading(
    plugin, "topic: " + topic + "\nmessage_type: visualization_msgs/msg/MarkerArray\n");

  YAML::Node saved = SaveAfterLoading(plugin, "topic: " + topic + "\n");

  EXPECT_EQ("visualization_msgs/msg/MarkerArray", saved["message_type"].as<std::string>(""));
}

TEST(MarkerType, ReportsAnUnsupportedPublishedType)
{
  const std::string topic = "/test_marker_type/unsupported";
  auto publisher = g_node->create_publisher<std_msgs::msg::String>(topic, 1);

  ErrorCapture<mapviz_plugins::MarkerPlugin> plugin;
  plugin.SetNode(*g_node);
  LoadTopic(plugin, "topic", topic);

  // Reported on connecting, or from the 1 s timer if discovery is slower.
  const auto published_as_string = [&plugin] {
      for (const std::string & error : plugin.errors) {
        if (error.find("is published as std_msgs/msg/String") != std::string::npos) {
          return true;
        }
      }
      return false;
    };
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (!published_as_string() && std::chrono::steady_clock::now() < deadline) {
    QCoreApplication::processEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(published_as_string());
  EXPECT_EQ(0u, g_node->count_subscribers(topic));
}

TEST(MarkerType, ReportsATopicWithAnUnknownSubstitution)
{
  // Name validation accepts "{...}", but expanding an unknown substitution throws.
  MarkerStatus plugin;
  plugin.SetNode(*g_node);

  EXPECT_NO_THROW(LoadTopic(plugin, "topic", "/test_marker_type/{unknown}"));
  EXPECT_EQ(0u, plugin.status.rfind("error: Invalid topic name", 0)) << plugin.status;

  // The timer must not throw either.
  EXPECT_TRUE(ProcessEventsFor(std::chrono::milliseconds(1500)));
  EXPECT_EQ(1, plugin.errors);
}

TEST(MarkerType, ClearsTheUnsupportedTypeErrorOnceAMarkerIsPublished)
{
  const std::string topic = "/test_marker_type/unsupported_then_marker";
  auto string_publisher = g_node->create_publisher<std_msgs::msg::String>(topic, 1);

  MarkerStatus plugin;
  plugin.SetNode(*g_node);
  LoadTopic(plugin, "topic", topic);
  ASSERT_TRUE(
    ProcessEventsUntil(
      [&plugin] {return plugin.status.find("is published as std_msgs") != std::string::npos;},
      std::chrono::seconds(3)));

  string_publisher.reset();
  auto marker_publisher = g_node->create_publisher<visualization_msgs::msg::Marker>(topic, 1);
  EXPECT_TRUE(
    ProcessEventsUntil(
      [&plugin] {return plugin.status == "warning: No messages received.";},
      std::chrono::seconds(3))) << plugin.status;
}

/// Loads a marker display with an invalid QoS @p setting; expects it reported once.
void ExpectInvalidQosReportedOnce(const std::string & setting)
{
  MarkerStatus plugin;
  plugin.SetNode(*g_node);
  EXPECT_NO_THROW(
    plugin.LoadConfigPlugin(
      YAML::Load(
        "topic: /test_marker_type/invalid_qos\n"
        "message_type: visualization_msgs/msg/Marker\n" + setting + "\n"), ""));
  EXPECT_EQ(0u, plugin.status.rfind("error: Unable to subscribe", 0)) << plugin.status;

  EXPECT_TRUE(ProcessEventsFor(std::chrono::milliseconds(1500)));
  EXPECT_EQ(1, plugin.errors) << plugin.status;
}

TEST(MarkerType, DoesNotRetryAnInvalidQosHistory)
{
  ExpectInvalidQosReportedOnce("qos_history: 9");  // rejected by rclcpp
}

TEST(MarkerType, DoesNotRetryAnInvalidQosReliability)
{
  ExpectInvalidQosReportedOnce("qos_reliability: 9");  // rejected by rcl
}

TEST(MarkerType, ReportsAnInvalidQosOnATopicAnotherDisplayLeft)
{
  // The other display's released subscription buys one retry, then the error is shown.
  const std::string topic = "/test_marker_type/left_behind";
  {
    mapviz_plugins::MarkerPlugin first;
    first.SetNode(*g_node);
    first.LoadConfigPlugin(
      YAML::Load("topic: " + topic + "\nmessage_type: visualization_msgs/msg/Marker\n"), "");
  }
  MarkerStatus second;
  second.SetNode(*g_node);
  second.LoadConfigPlugin(
    YAML::Load(
      "topic: " + topic + "\nmessage_type: visualization_msgs/msg/MarkerArray\n"
      "qos_reliability: 9\n"), "");

  EXPECT_TRUE(ProcessEventsFor(std::chrono::milliseconds(2500)));
  EXPECT_EQ(0u, second.status.rfind("error: Unable to subscribe", 0)) << second.status;
  EXPECT_EQ(1, second.errors);
}

TEST(MarkerType, WaitsForAnotherDisplayToReleaseTheTopic)
{
  // No publisher, so neither display has a reason to switch type: the second waits rather
  // than ask the middleware for a second type on the topic (which Fast DDS refuses). It
  // names the topic relative to the node's namespace, but it is the same topic.
  const std::string topic = "/test_marker_type/held_by_another";
  MarkerStatus first;
  first.SetNode(*g_node);
  first.LoadConfigPlugin(
    YAML::Load("topic: " + topic + "\nmessage_type: visualization_msgs/msg/Marker\n"), "");
  MarkerStatus second;
  second.SetNode(*g_node);
  EXPECT_NO_THROW(
    second.LoadConfigPlugin(
      YAML::Load(
        "topic: " + topic.substr(1) + "\nmessage_type: visualization_msgs/msg/MarkerArray\n"),
      ""));
  EXPECT_EQ(0u, second.status.rfind("warning: Waiting to subscribe", 0)) << second.status;
  EXPECT_EQ(0, second.errors);

  first.LoadConfigPlugin(YAML::Load("topic: /test_marker_type/elsewhere\n"), "");
  EXPECT_TRUE(
    ProcessEventsUntil(
      [&topic] {
        const auto subscriptions = g_node->get_subscriptions_info_by_topic(topic);
        return subscriptions.size() == 1 &&
               subscriptions.front().topic_type() == "visualization_msgs/msg/MarkerArray";
      },
      std::chrono::seconds(3)));
  EXPECT_EQ("warning: No messages received.", second.status);
  EXPECT_EQ(0, second.errors);
}

TEST(MarkerType, ReportsAnUnsupportedSavedTypeWhileAnotherDisplayHoldsTheTopic)
{
  const std::string topic = "/test_marker_type/held_unsupported";
  mapviz_plugins::MarkerPlugin first;
  first.SetNode(*g_node);
  first.LoadConfigPlugin(
    YAML::Load("topic: " + topic + "\nmessage_type: visualization_msgs/msg/Marker\n"), "");
  MarkerStatus second;
  second.SetNode(*g_node);
  second.LoadConfigPlugin(
    YAML::Load("topic: " + topic + "\nmessage_type: std_msgs/msg/String\n"), "");
  EXPECT_EQ(0u, second.status.rfind("error: Unable to subscribe", 0)) << second.status;
}

TEST(MarkerType, ClearsTheUnsupportedTypeErrorWhenThePublisherGoes)
{
  const std::string topic = "/test_marker_type/unsupported_then_gone";
  auto string_publisher = g_node->create_publisher<std_msgs::msg::String>(topic, 1);

  MarkerStatus plugin;
  plugin.SetNode(*g_node);
  LoadTopic(plugin, "topic", topic);
  ASSERT_TRUE(
    ProcessEventsUntil(
      [&plugin] {return plugin.status.find("is published as std_msgs") != std::string::npos;},
      std::chrono::seconds(3)));

  string_publisher.reset();
  EXPECT_TRUE(
    ProcessEventsUntil(
      [&plugin] {return plugin.status == "warning: No messages received.";},
      std::chrono::seconds(3))) << plugin.status;
}

TEST(MarkerType, SavesNoTypeWithoutATopic)
{
  mapviz_plugins::MarkerPlugin plugin;
  plugin.SetNode(*g_node);
  YAML::Node saved =
    SaveAfterLoading(plugin, "message_type: visualization_msgs/msg/MarkerArray\n");
  EXPECT_FALSE(saved["message_type"]);
}

TEST(MarkerType, JoinsTheTypeAnotherDisplayHoldsWhenBothArePublished)
{
  // Each publisher gets a context of its own, as Fast DDS refuses two types on one topic
  // within a participant.
  const std::string topic = "/test_marker_type/both_types";
  auto marker_context = std::make_shared<rclcpp::Context>();
  marker_context->init(0, nullptr);
  auto marker_node = std::make_shared<rclcpp::Node>(
    "test_marker_publisher", rclcpp::NodeOptions().context(marker_context));
  auto marker_publisher =
    marker_node->create_publisher<visualization_msgs::msg::Marker>(topic, 10);
  auto array_context = std::make_shared<rclcpp::Context>();
  array_context->init(0, nullptr);
  auto array_node = std::make_shared<rclcpp::Node>(
    "test_marker_array_publisher", rclcpp::NodeOptions().context(array_context));
  auto array_publisher =
    array_node->create_publisher<visualization_msgs::msg::MarkerArray>(topic, 10);
  ASSERT_TRUE(
    ProcessEventsUntil(
      [&topic] {return g_node->count_publishers(topic) == 2;}, std::chrono::seconds(10)));

  mapviz_plugins::MarkerPlugin first;
  first.SetNode(*g_node);
  first.LoadConfigPlugin(
    YAML::Load("topic: " + topic + "\nmessage_type: visualization_msgs/msg/Marker\n"), "");
  MarkerStatus second;
  second.SetNode(*g_node);
  EXPECT_NO_THROW(
    second.LoadConfigPlugin(
      YAML::Load("topic: " + topic + "\nmessage_type: visualization_msgs/msg/MarkerArray\n"),
      ""));

  EXPECT_TRUE(
    ProcessEventsUntil(
      [&marker_publisher] {return marker_publisher->get_subscription_count() == 2;},
      std::chrono::seconds(10))) << second.status;

  marker_publisher.reset();
  array_publisher.reset();
  marker_node.reset();
  array_node.reset();
  marker_context->shutdown("test finished");
  array_context->shutdown("test finished");
}

TEST(MarkerType, SavesNoTypeForATopicStillBeingTypedIn)
{
  mapviz_plugins::MarkerPlugin plugin;
  plugin.SetNode(*g_node);
  plugin.LoadConfigPlugin(
    YAML::Load(
      "topic: /test_marker_type/committed\n"
      "message_type: visualization_msgs/msg/MarkerArray\n"), "");

  auto * topic_box = plugin.GetConfigWidget(nullptr)->findChild<QLineEdit *>("topic");
  ASSERT_NE(nullptr, topic_box);
  topic_box->setText("/test_marker_type/being_typed");

  YAML::Emitter emitter;
  emitter << YAML::BeginMap;
  plugin.SaveConfigPlugin(emitter, "");
  emitter << YAML::EndMap;
  YAML::Node saved = YAML::Load(emitter.c_str());
  EXPECT_EQ("/test_marker_type/being_typed", saved["topic"].as<std::string>(""));
  EXPECT_FALSE(saved["message_type"]);
}

TEST(MarkerType, ResubscribesWhenAReloadChangesOnlyTheType)
{
  const std::string topic = "/test_marker_type/type_only_reload";
  mapviz_plugins::MarkerPlugin plugin;
  plugin.SetNode(*g_node);
  plugin.LoadConfigPlugin(
    YAML::Load("topic: " + topic + "\nmessage_type: visualization_msgs/msg/MarkerArray\n"),
    "");

  plugin.LoadConfigPlugin(YAML::Load("message_type: visualization_msgs/msg/Marker\n"), "");

  EXPECT_TRUE(
    ProcessEventsUntil(
      [&topic] {
        const auto subscriptions = g_node->get_subscriptions_info_by_topic(topic);
        return subscriptions.size() == 1 &&
               subscriptions.front().topic_type() == "visualization_msgs/msg/Marker";
      },
      std::chrono::seconds(3)));
}

TEST(InvalidNames, MarkerReportsARejectedTopic)
{
  // The marker plugin looks the topic's publishers up before it subscribes, so the name is
  // rejected there rather than in Subscribe().
  ErrorCapture<mapviz_plugins::MarkerPlugin> plugin;
  plugin.SetNode(*g_node);

  EXPECT_NO_THROW(LoadTopic(plugin, "topic", "/test_invalid_names/not a marker topic"));

  EXPECT_TRUE(ReportedInvalid(plugin.errors, "topic name"));
}

int main(int argc, char ** argv)
{
  // The plugins build QWidget based config panels in their constructors, so a
  // QApplication has to exist first.  The test runs headless; see the ENV set
  // on this target in CMakeLists.txt.
  QApplication app(argc, argv);
  rclcpp::init(argc, argv);
  g_node = std::make_shared<rclcpp::Node>("test_plugin_config");
  g_canvas = std::make_unique<QOpenGLWidget>();

  testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();

  g_canvas.reset();
  g_node.reset();
  rclcpp::shutdown();
  return result;
}
